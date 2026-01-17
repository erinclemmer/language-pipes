import os
import json
import torch
from typing import List, Tuple, Optional

from transformers.models.auto import AutoConfig
from llm_layer_collector.auto.auto_rms import AutoRMSNorm
from transformers.configuration_utils import PretrainedConfig
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

from language_pipes.util import size_of_tensor, tensor_hash
from language_pipes.util.enums import ModelPartType

def get_size_of_layer(config: PretrainedConfig, layer_idx: int) -> Tuple[float, str]:
    print(f"Calculating layer size for layer {layer_idx}...")
    print("Calculating attention size...")
    lyr = AutoDecoderLayer(config, layer_idx).cls.to(dtype=torch.float16)
    attn_tensors = []
    try:
        attn_tensors.append(lyr.input_layernorm.weight)
    except AttributeError:
        pass

    try:
        attn_tensors.append(lyr.post_attention_layernorm.weight)
    except AttributeError:
        pass

    try:
        attn_tensors.extend([
            lyr.self_attn.q_norm.weight,
            lyr.self_attn.k_norm.weight
        ])
    except AttributeError:
        pass

    attn_tensors.extend([
        lyr.self_attn.q_proj.weight,
        lyr.self_attn.k_proj.weight,
        lyr.self_attn.v_proj.weight,
        lyr.self_attn.o_proj.weight
    ])

    hash = tensor_hash(lyr.self_attn.q_proj.weight)
    attn_size = sum([size_of_tensor(t) for t in attn_tensors])
    print(f"Attention size is {attn_size / 10**6:.2f}MB")

    print("Calculating mlp size...")
    mlp_tensors = []
    try:
        mlp_tensors = [
            lyr.mlp.gate_proj.weight,
            lyr.mlp.up_proj.weight,
            lyr.mlp.down_proj.weight
        ] 
    except AttributeError:
        for i in range(0, lyr.mlp.num_experts):
            mlp_tensors.extend([
                lyr.mlp.gate.weight,
                lyr.mlp.experts[i].gate_proj.weight,
                lyr.mlp.experts[i].up_proj.weight,
                lyr.mlp.experts[i].down_proj.weight
            ]) 

    mlp_size = sum([size_of_tensor(t) for t in mlp_tensors])
    print(f"MLP Size is {mlp_size / 10**6:.2f}MB")

    total_size = attn_size + mlp_size

    print(f"Total Layer Size is {total_size / 10**6:.2f}MB")

    return total_size, hash


def get_avg_layer_size(model_path: str) -> Tuple[int, List[str]]:
    if not os.path.exists(model_path):
        print(f'Model {model_path} not found')
        return -1, []
    config = AutoConfig.from_pretrained(model_path)

    total_size = 0
    layer_hashes = []
    for size, hash in [get_size_of_layer(config, i) for i in range(config.num_hidden_layers)]:
        total_size += size
        layer_hashes.append(hash)
    
    avg_layer_size = total_size / config.num_hidden_layers
    layer_hashes = layer_hashes
    
    return avg_layer_size, layer_hashes

def data_of_type(typ: ModelPartType, model_path: str) -> Tuple[float, str]:
    config = AutoConfig.from_pretrained(model_path)
    
    size = 0
    hash = ''
    if typ == ModelPartType.EMBED:
        print("Calculating embedding size...")
        e  = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(dtype=torch.float16)
        size = size_of_tensor(e.weight)
        hash = tensor_hash(e.weight)
        print(f"Embedding is {size / 10**6:.2f}MB")
        
    if typ == ModelPartType.NORM:
        n = AutoRMSNorm(config).to(dtype=torch.float16)
        size = size_of_tensor(n.cls.weight)
        hash = tensor_hash(n.cls.weight)
    if typ == ModelPartType.HEAD:
        print("Calculating head size...")
        h = torch.nn.Linear(config.hidden_size, config.vocab_size).to(dtype=torch.float16)
        size = size_of_tensor(h.weight)
        hash = tensor_hash(h.weight)
        print(f"Head is {size / 10**6:.2f}MB")
    
    return size, hash

def get_computed_data(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model {model_path} not found')
    computed_path = os.path.join(model_path, 'meta_data.json')
    if os.path.exists(computed_path):
        with open(computed_path) as f:
            return json.load(f)

    print('Computing Data for ' + model_path)
    meta_data = { }
    model_path = os.path.join(model_path, 'data')
    size, hash = data_of_type(ModelPartType.EMBED, model_path)
    meta_data['embed_size'] = size
    meta_data['embed_hash'] = hash
    size, hash = data_of_type(ModelPartType.NORM, model_path)
    size, hash = data_of_type(ModelPartType.HEAD, model_path)
    meta_data['head_size'] = size
    meta_data['head_hash'] = hash
    size, hash = get_avg_layer_size(model_path)
    meta_data['avg_layer_size'] = size
    meta_data['layer_hashes'] = hash

    with open(computed_path, 'w') as f:
        json.dump(meta_data, f)

    return meta_data

class LlmMetadata:
    embed_size: int
    head_size: int
    avg_layer_size: int
    
    embed_hash: str
    head_hash: str
    layer_hashes: List[str]

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            return
        data = get_computed_data(model_dir)
        self.embed_size = data['embed_size']
        self.head_size = data['head_size']
        self.avg_layer_size = data['avg_layer_size']
        self.embed_hash = data['embed_hash']
        self.head_hash = data['head_hash']
        self.layer_hashes = data['layer_hashes']

    def to_json(self):
        return {
            'embed_size': self.embed_size,
            'head_size': self.head_size,
            'avg_layer_size': self.avg_layer_size,
            'embed_hash': self.embed_hash,
            'head_hash': self.head_hash,
            'layer_hashes': self.layer_hashes
        }

    @staticmethod
    def from_dict(data: dict) -> 'LlmMetadata':
        c = LlmMetadata(None)
        c.embed_size = data['embed_size']
        c.head_size = data['head_size']
        c.avg_layer_size = data['avg_layer_size']
        c.embed_hash = data['embed_hash']
        c.head_hash = data['head_hash']
        c.layer_hashes = data['layer_hashes']
        return c
    
def validate_model(c1: LlmMetadata, c2: LlmMetadata):
    return c1.embed_hash == c2.embed_hash and c1.head_hash == c2.head_hash and c1.layer_hashes == c2.layer_hashes
