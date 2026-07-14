from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from llm_layer_collector.state_obj import LLmComputationState

def has_sliding_window(config: PretrainedConfig) -> bool:
    try:
        return config.sliding_window is not None
    except:  # noqa: E722
        return False

class Qwen3MoeModel:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig,
        mask_kwargs: Any
    ) -> LLmComputationState:
        state.causal_mask = {
            "full_attention": create_causal_mask(**mask_kwargs)
        }
        
        if has_sliding_window(config):
            state.causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        state.position_embeddings["full_attention"] = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        config: PretrainedConfig,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        layer_type = "sliding_attention" if has_sliding_window(config) else "full_attention"
        kwargs = { # pyright: ignore[reportUnknownVariableType]
            "hidden_states": state.state,
            "attention_mask": state.causal_mask[layer_type], # type: ignore
            "position_embeddings": state.position_embeddings["full_attention"],
            "position_ids": state.position_ids,
            "past_key_values": cache
        }
        
        return layer.cls(**kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        