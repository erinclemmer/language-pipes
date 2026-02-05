from typing import Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from language_pipes.llm_layer_collector.state_obj import LLmComputationState

mapper = {
    "llama": LlamaDecoderLayer,
    "qwen3": Qwen3DecoderLayer,
    "gemma3_text": Gemma3DecoderLayer,
    "qwen3_moe": Qwen3MoeDecoderLayer
}

def getClass(config: PretrainedConfig) -> GradientCheckpointingLayer:
    return mapper[config.model_type]

class AutoDecoderLayer:
    def __init__(self, config: PretrainedConfig, layer_index: int):
        self.config = config
        if self.config._attn_implementation is None:
            self.config._attn_implementation = "eager"
        self.cls = getClass(self.config)(self.config, layer_index)

    def to_empty(self, device: Optional[str]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: str):
        self.cls = self.cls.to(device)
        return self