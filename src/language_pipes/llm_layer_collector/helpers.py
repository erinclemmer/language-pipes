import os
from typing import Dict, List

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.models.glm4v.modeling_glm4v import Glm4vTextModel
from transformers.models.glm4v.configuration_glm4v import Glm4vTextConfig

# Fix bug in transformers 4.57
Glm4vTextModel.config_class = Glm4vTextConfig
AutoModelForCausalLM.register(Glm4vTextConfig, Glm4vTextModel)

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

def get_config(model_dir: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type == "glm4v":
        config = Glm4vTextConfig.from_json_file(model_dir + "/config.json")
        # Some GLM-4.1V checkpoints ship with rope_scaling unset, but
        # Glm4vTextAttention.forward unconditionally reads
        # rope_scaling["mrope_section"]. Provide a compatible default.
        if config.rope_scaling is None:
            # For GLM-4.1V-9B, head_dim is 128 and the HF implementation
            # expects a 3-way split that sums to 64 before doubling.
            config.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
            }

    with torch.device('meta'):
        meta_model = AutoModelForCausalLM.from_config(config)
    return meta_model.config
