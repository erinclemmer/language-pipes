import os
from typing import Dict, List

import torch
from safetensors import safe_open

def get_shard_keys(st: safe_open) -> List[str]:
    return st.keys() # type: ignore

def get_shard_tensor(st: safe_open, name: str) -> torch.Tensor:
    return st.get_tensor(name) # type: ignore

def load_shard_tensor(
        layer_file_cache: Dict[str, str], 
        model_dir: str,
        layer_name: str, 
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
    if layer_name not in layer_file_cache:
        raise ValueError(f'Could not find layer file for layer {layer_name}')
    file = layer_file_cache[layer_name]
    shard = safe_open(os.path.join(model_dir, file), framework='pt', device=str(device))
    return get_shard_tensor(shard, layer_name).to(dtype)