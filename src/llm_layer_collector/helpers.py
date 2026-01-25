import os

import torch
from safetensors import safe_open

def load_shard_tensor(
        layer_file_cache: dict, 
        model_dir: str,
        layer_name: str, 
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
    if layer_name not in layer_file_cache:
        raise ValueError(f'Could not find layer file for layer {layer_name}')
    file = layer_file_cache[layer_name]
    shard: dict = safe_open(os.path.join(model_dir, file), framework='pt', device=device)
    return shard.get_tensor(layer_name).to(dtype)