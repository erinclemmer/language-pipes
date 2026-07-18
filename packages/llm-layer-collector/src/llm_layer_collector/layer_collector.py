import os
import gc
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextScaledWordEmbedding

from llm_layer_collector.helpers import get_config
from llm_layer_collector.modeling.Gemma4Model import Gemma4PerLayerEmbedder
from llm_layer_collector.load_layer import load_layers
from llm_layer_collector.cache import build_cache_data
from llm_layer_collector.helpers import load_shard_tensor
from llm_layer_collector.auto.auto_rms import AutoRMSNorm
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer


class LlmLayerCollector:
    layer_prefix: str
    norm_layer_name: str
    input_embedding_layer_name: str
    lm_head_name: str
    shard_pattern: str

    config: PretrainedConfig

    model_dir: Path
    cache_file: Path

    num_layers: int
    num_shards: int
    dtype: torch.dtype
    device: torch.device
    load_in_8bit: bool
    layer_files: Dict[str, str]

    def __init__(
        self,
        model_dir: Path,
        cache_file: Optional[Path] = None,
        shard_pattern: str = r"model-(\d+)-of-(\d+).safetensors",
        layer_prefix: str = "model.layers.",
        input_embedding_layer_name: str = "model.embed_tokens.weight",
        norm_layer_name: str = "model.norm.weight",
        lm_head_name: str = "lm_head.weight",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        load_in_8bit: bool = False,
    ):
        config_file_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file_path):
            raise FileNotFoundError("Could not find config file " + config_file_path)

        config = get_config(model_dir)
        self.config = config

        if "_attn_implementation" not in self.config:
            self.config._attn_implementation = "sdpa"  # pyright: ignore[reportPrivateUsage]
        self.num_layers = self.config.num_hidden_layers

        self.model_dir = model_dir

        self.lm_head_name = lm_head_name
        self.layer_prefix = layer_prefix
        self.norm_layer_name = norm_layer_name
        self.input_embedding_layer_name = input_embedding_layer_name
        self.shard_pattern = shard_pattern

        self.load_in_8bit = load_in_8bit
        # bitsandbytes LLM.int8 kernels compute in fp16, so the unquantized
        # pieces (embedding, norms, head) must match.
        self.dtype = torch.float16 if load_in_8bit else dtype
        self.device = device
        self.layer_files = {}
        if cache_file is None:
            raise Exception("Must provide cache file path")
        self.cache_file = cache_file
        self._read_cache()

        # Multimodal checkpoints nest the head (e.g. "language_model.lm_head.weight");
        # without this, load_head would silently fall back to the embedding weights
        # even for untied models.
        if self.lm_head_name not in self.layer_files:
            for key in self.layer_files:
                if key.endswith("lm_head.weight"):
                    self.lm_head_name = key
                    break

    def _read_cache(self):
        if not os.path.exists(self.cache_file):
            return self._build_cache()
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.layer_files = data["layer_files"]
            self.layer_prefix = data["layer_prefix"]
            self.input_embedding_layer_name = data["input_embed_name"]
            self.norm_layer_name = data["norm_name"]

    # Use model.safetensors.index.json by default if it exists
    def _build_cache(self):
        self.layer_files = build_cache_data(
            self.model_dir, self.shard_pattern, self.device
        )

        for key in self.layer_files.keys():
            if "layers.0" in key and "vision_tower" not in key:
                self.layer_prefix = key.split("layers.0")[0] + "layers."
            if key.endswith("embed_tokens.weight"):
                self.input_embedding_layer_name = key

        derived_norm = self.input_embedding_layer_name.replace(
            "embed_tokens.weight", "norm.weight"
        )
        if derived_norm in self.layer_files:
            self.norm_layer_name = derived_norm

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "layer_files": self.layer_files,
                "layer_prefix": self.layer_prefix,
                "input_embed_name": self.input_embedding_layer_name,
                "norm_name": self.norm_layer_name
            }, f, indent=4)

    def _load_shard_tensor(self, layer_name: str, device: torch.device) -> torch.Tensor:
        return load_shard_tensor(
            self.layer_files, self.model_dir, layer_name, device, self.dtype
        )

    def load_input_embedding(
        self, device: Optional[torch.device] = None
    ) -> torch.nn.Embedding:
        device = self.device if device is None else device
        emb_weight = self._load_shard_tensor(self.input_embedding_layer_name, device)

        if self.config.model_type == "gemma3_text":
            padding_idx = (
                0 if self.config.pad_token_id is None else self.config.pad_token_id
            )
            embed = Gemma3TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                padding_idx,
                embed_scale=self.config.hidden_size**0.5,
            )
            embed.weight = torch.nn.Parameter(emb_weight)
            return embed.to(device=device, dtype=self.dtype)

        if self.config.model_type in ("gemma4_text", "gemma4_unified_text"):
            padding_idx = (
                0 if self.config.pad_token_id is None else self.config.pad_token_id
            )
            embed_cls = (
                Gemma4TextScaledWordEmbedding
                if self.config.model_type == "gemma4_text"
                else Gemma4UnifiedTextScaledWordEmbedding
            )
            embed = embed_cls(
                self.config.vocab_size,
                self.config.hidden_size,
                padding_idx,
                embed_scale=self.config.hidden_size**0.5,
            )
            embed.weight = torch.nn.Parameter(emb_weight)
            return embed.to(device=device, dtype=self.dtype)

        return torch.nn.Embedding.from_pretrained(emb_weight)  # pyright: ignore[reportUnknownMemberType]

    def load_per_layer_embedder(
        self, device: Optional[torch.device] = None
    ) -> Optional[Gemma4PerLayerEmbedder]:
        """Load the three Gemma4 Per-Layer Embedding (PLE) weights.

        These live only on the head/embedding node — ``embed_tokens_per_layer`` is the
        largest single tensor in the checkpoint, so it must never be loaded on layer
        nodes. Returns ``None`` for models that don't use PLE.
        """
        if self.config.model_type != "gemma4_text" or not getattr(
            self.config, "hidden_size_per_layer_input", 0
        ):
            return None

        device = self.device if device is None else device
        embedder = Gemma4PerLayerEmbedder(self.config)

        embedder.embed_tokens_per_layer.weight = torch.nn.Parameter(
            self._load_shard_tensor("model.language_model.embed_tokens_per_layer.weight", device)
        )
        embedder.per_layer_model_projection.weight = torch.nn.Parameter(
            self._load_shard_tensor("model.language_model.per_layer_model_projection.weight", device)
        )
        embedder.per_layer_projection_norm.weight = torch.nn.Parameter(
            self._load_shard_tensor("model.language_model.per_layer_projection_norm.weight", device)
        )

        return embedder.to(device=device, dtype=self.dtype)

    def load_norm(self, device: Optional[torch.device] = None) -> AutoRMSNorm:
        device = self.device if device is None else device
        norm = AutoRMSNorm(self.config)
        norm.cls.weight = torch.nn.Parameter(
            self._load_shard_tensor(self.norm_layer_name, device)
        )
        return norm

    def load_head(self, device: Optional[torch.device] = None) -> torch.nn.Linear:
        device = self.device if device is None else device
        weight = None

        if self.lm_head_name not in self.layer_files:
            weight = self.load_input_embedding(device).weight
        else:
            weight = self._load_shard_tensor(self.lm_head_name, device)

        # Decoder-only LMs in this project use an lm_head without bias.
        head = torch.nn.Linear(
            weight.size()[1],
            weight.size()[0],
            bias=False,
            device=device,
            dtype=self.dtype,
        )
        head.weight = torch.nn.Parameter(weight)
        return head

    def load_layer_set(
        self, start_layer: int, end_layer: int, device: Optional[torch.device] = None
    ) -> List[AutoDecoderLayer]:
        device = self.device if device is None else device
        layers: List[AutoDecoderLayer] = []
        for i in range(start_layer, end_layer + 1, 3):
            layers.extend(
                load_layers(
                    min(i, end_layer),
                    min(i + 2, end_layer),
                    self.layer_prefix,
                    self.layer_files,
                    self.config,
                    self.model_dir,
                    device,
                    self.dtype,
                    self.load_in_8bit,
                )
            )
        gc.collect()
        return layers
