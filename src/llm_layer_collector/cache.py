import re
import os
from typing import List

from safetensors import safe_open

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
        device: str
    ):
    layer_files = { }
    for file in get_shard_files(shard_pattern, model_dir):
        full_path = os.path.join(model_dir, file)
        shard: dict = safe_open(full_path, framework='pt', device=device)
        for key in shard.keys():
            layer_files[key] = file
        del shard

    return layer_files