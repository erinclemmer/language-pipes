import gc
import torch
from typing import List, Dict

from safetensors import safe_open
from transformers.configuration_utils import PretrainedConfig
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

def files_to_load_for_layer(
        layer_prefix: str,
        layer_file_cache: dict,
    ) -> List[str]:
    files_to_load = []
    for key in layer_file_cache.keys():
        if key.startswith(layer_prefix) and layer_file_cache[key] not in files_to_load:
            files_to_load.append(layer_file_cache[key])
    if len(files_to_load) == 0:
        raise Exception("Could not find layer data for layer prefix " + layer_prefix)
    return files_to_load

def files_to_load_for_layers(
        start_layer: int,
        end_layer: int,
        layer_prefix: str,
        layer_file_cache: dict
    ) -> List[str]:
    files_to_load = []
    for i in range(start_layer, end_layer+1):
        for f in files_to_load_for_layer(f'{layer_prefix}{i}.', layer_file_cache):
            if f not in files_to_load:
                files_to_load.append(f)
    return files_to_load

def get_shard_data(
        start_layer: int,
        end_layer: int,
        device: str,
        model_dir: str,
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        dtype: str
    ) -> Dict[str, torch.Tensor]:
    prefixes = [f'{layer_prefix}{i}.' for i in range(start_layer, end_layer+1)]
    shard_data = { }
    for file_path in files_to_load_for_layers(start_layer, end_layer, layer_prefix, layer_file_cache):
        full_path = f'{model_dir}/{file_path}'
        shard: dict = safe_open(full_path, framework='pt', device=device)
        for key in shard.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    shard_data[key] = shard.get_tensor(key).detach().to(dtype)
        del shard
        gc.collect()
    
    return shard_data

def load_layer(
        config: PretrainedConfig, 
        idx: int, 
        shard_data: Dict,
        layer_prefix: str,
        device: str,
        dtype: str
    ) -> AutoDecoderLayer:
    torch.set_default_device('meta')
    lyr = AutoDecoderLayer(config, idx)
    torch.set_default_device(device)
    lyr = lyr.to_empty(device=device)
    for key in shard_data.keys():
        if key.startswith(f'{layer_prefix}{idx}.'):
            module_name = key.replace(f'{layer_prefix}{idx}.', '').replace('.weight', '')
            module = lyr.get_submodule(module_name)
            module.load_state_dict({ "weight": shard_data[key] })

    return lyr.to(dtype)

def load_layers(
        start_layer: int, 
        end_layer: int, 
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        config: PretrainedConfig,
        model_dir: str,
        device: str,
        dtype: str
    ) -> List[AutoDecoderLayer]:
    torch.set_default_device(device)
    shard_data = get_shard_data(start_layer, end_layer, device, model_dir, layer_prefix, layer_file_cache, dtype)
    layers = []
    for i in range(start_layer, end_layer+1):
        layers.append(load_layer(config, i, shard_data, layer_prefix, device, dtype))

    torch.set_default_device('cpu')
    return layers
