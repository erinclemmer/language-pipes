import torch
from typing import Optional

from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

mapper = { # type: ignore
    "llama": LlamaDecoderLayer,
    "qwen3": Qwen3DecoderLayer,
    "gemma3_text": Gemma3DecoderLayer,
    "qwen3_moe": Qwen3MoeDecoderLayer
}

def getClass(config: PretrainedConfig) -> GradientCheckpointingLayer:
    return mapper[config.model_type] # type: ignore

class AutoDecoderLayer:
    cls: GradientCheckpointingLayer

    def __init__(self, config: PretrainedConfig, layer_index: int):
        self.config = config
        if self.config._attn_implementation_internal is None: # type: ignore
            self.config._attn_implementation = "eager" # type: ignore
        self.cls = getClass(self.config)(self.config, layer_index)

    def to_empty(self, device: Optional[torch.device]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: torch.device, dtype: torch.dtype) -> 'AutoDecoderLayer':
        self.cls = self.cls.to(device=device, dtype=dtype)
        return self