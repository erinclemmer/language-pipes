from typing import Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from llm_layer_collector.state_obj import LLmComputationState

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

    def __call__(
        self, 
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        attention_type = "full_attention"
        try:
            attention_type = self.cls.attention_type
        except AttributeError:
            pass

        try:
            if self.config.sliding_window is not None:
                attention_type = "sliding_attention"
        except AttributeError:
            pass

        kwargs = {
            "hidden_states": state.state,
            "attention_mask": state.causal_mask[attention_type],
            "position_ids": state.position_ids,
            "use_cache": self.config.use_cache,
            "cache_position": state.cache_position,
        }

        if self.config.model_type == "qwen3" or self.config.model_type == "qwen3_moe":
            kwargs["past_key_values"] = cache
        else:
            kwargs["past_key_value"] = cache

        if self.config.model_type == "gemma3_text":
            kwargs["position_embeddings_local"] = state.position_embeddings_local
            kwargs["position_embeddings_global"] = state.position_embeddings_global
            return self.cls(**kwargs)[0]
        else:
            kwargs["position_embeddings"] = state.position_embeddings
            return self.cls(**kwargs)

    def to_empty(self, device: Optional[str]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: str):
        self.cls = self.cls.to(device)
        return self