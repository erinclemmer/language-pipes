import re
import os
import torch
from typing import List, Dict

from safetensors import safe_open
from language_pipes.llm_layer_collector.helpers import get_shard_keys

def get_shard_files(shard_pattern: str, model_dir: str) -> List[str]:
    if 'model.safetensors' in os.listdir(model_dir):
        return ['model.safetensors']
    
    multiple_pattern = re.compile(shard_pattern)
    shard_files = [f for f in os.listdir(model_dir) if multiple_pattern.match(f)]
    if not shard_files:
        raise Exception("No Shard files in specified directory " + model_dir)

    shard_files.sort()
    return shard_files

def build_cache_data(
        model_dir: str,
        shard_pattern: str,
        device: torch.device
    ) -> Dict[str, str]:
    layer_files: Dict[str, str] = { }
    for file in get_shard_files(shard_pattern, model_dir):
        full_path = os.path.join(model_dir, file)
        shard = safe_open(full_path, framework='pt', device=str(device))
        for key in get_shard_keys(shard):
            layer_files[key] = file
        del shard

    return layer_files