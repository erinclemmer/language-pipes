import os
import gc
import json
import tqdm
from typing import List, Dict, Optional

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

from language_pipes.llm_layer_collector.helpers import get_config
from language_pipes.llm_layer_collector.load_layer import load_layers
from language_pipes.llm_layer_collector.cache import build_cache_data
from language_pipes.llm_layer_collector.helpers import load_shard_tensor
from language_pipes.llm_layer_collector.auto.auto_rms import AutoRMSNorm
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer

class LlmLayerCollector:
    layer_prefix: str
    norm_layer_name: str
    input_embedding_layer_name: str
    lm_head_name: str
    shard_pattern: str
    
    config: PretrainedConfig
    
    model_dir: str
    cache_file: str

    num_layers: int
    num_shards: int
    dtype: torch.dtype
    device: torch.device
    layer_files: Dict[str, str]

    def __init__(
            self, 
            model_dir: str,
            cache_file: Optional[str] = None,
            shard_pattern: str = r'model-(\d+)-of-(\d+).safetensors',
            layer_prefix: str = 'model.layers.',
            input_embedding_layer_name: str = 'model.embed_tokens.weight',
            norm_layer_name: str = 'model.norm.weight',
            lm_head_name: str = 'lm_head.weight',
            dtype: torch.dtype = torch.float16,
            device: torch.device = torch.device('cpu')
        ):
        config_file_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError('Could not find config file ' + config_file_path)
        
        config = get_config(model_dir)
        self.config = config
        if "_attn_implementation" not in self.config:
            self.config._attn_implementation = "sdpa" # pyright: ignore[reportPrivateUsage]
        self.num_layers = self.config.num_hidden_layers
        
        self.model_dir = model_dir
        
        self.lm_head_name = lm_head_name
        self.layer_prefix = layer_prefix
        self.norm_layer_name = norm_layer_name
        self.input_embedding_layer_name = input_embedding_layer_name
        self.shard_pattern = shard_pattern

        self.dtype = dtype
        self.device = device
        self.layer_files = { }
        if cache_file is None:
            raise Exception("Must provide cache file path")
        self.cache_file = cache_file

        if not os.path.exists(self.cache_file):
            self._build_cache()
        else:
            self._read_cache()

    def _read_cache(self):
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError('Could not find cache file ' + self.cache_file)
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            self.layer_files = json.load(f)

    # Use model.safetensors.index.json by default if it exists
    def _build_cache(self):
        self.layer_files = build_cache_data(self.model_dir, self.shard_pattern, self.device)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.layer_files, f, indent=4)

    def _load_shard_tensor(self, layer_name: str, device: torch.device) -> torch.Tensor:
        return load_shard_tensor(self.layer_files, self.model_dir, layer_name, device, self.dtype)

    def load_input_embedding(self, device: Optional[torch.device] = None) -> torch.nn.Embedding:
        device = self.device if device is None else device
        emb_weight = self._load_shard_tensor(self.input_embedding_layer_name, device)

        if self.config.model_type == "gemma3_text":
            padding_idx = 0 if self.config.pad_token_id is None else self.config.pad_token_id
            embed = Gemma3TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                padding_idx,
                embed_scale=self.config.hidden_size ** 0.5
            )
            embed.weight = torch.nn.Parameter(emb_weight)
            return embed.to(device=device, dtype=self.dtype)

        return torch.nn.Embedding.from_pretrained(emb_weight) # pyright: ignore[reportUnknownMemberType]
    
    def load_norm(self, device: Optional[torch.device] = None) -> AutoRMSNorm:
        device = self.device if device is None else device
        norm = AutoRMSNorm(self.config)
        norm.cls.weight = torch.nn.Parameter(self._load_shard_tensor(self.norm_layer_name, device))
        return norm
    
    def load_head(self, device: Optional[torch.device] = None) -> torch.nn.Linear:
        device = self.device if device is None else device
        weight = None
        
        if self.lm_head_name not in self.layer_files:
            weight = self.load_input_embedding(device).weight
        else:
            weight = self._load_shard_tensor(self.lm_head_name, device)

        # Decoder-only LMs in this project use an lm_head without bias.
        head = torch.nn.Linear(weight.size()[1], weight.size()[0], bias=False, device=device, dtype=self.dtype)
        head.weight = torch.nn.Parameter(weight)
        return head

    def load_layer_set(self, start_layer: int, end_layer: int, device: Optional[torch.device] = None) -> List[AutoDecoderLayer]:
        device = self.device if device is None else device
        layers: List[AutoDecoderLayer] = []
        for i in tqdm.tqdm(range(start_layer, end_layer+1, 3)):
            layers.extend(load_layers(min(i, end_layer), min(i+2, end_layer), self.layer_prefix, self.layer_files, self.config, self.model_dir, device, self.dtype))
        gc.collect()
        return layers
