from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask
from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

class LlamaModel:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig,
        mask_kwargs: Any
    ) -> LLmComputationState:
        state.causal_mask["full_attention"] = create_causal_mask(**mask_kwargs) # type: ignore
        state.position_embeddings["full_attention"] = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        kwargs = { # pyright: ignore[reportUnknownVariableType]
            "hidden_states": state.state,
            "attention_mask": state.causal_mask["full_attention"],
            "position_embeddings": state.position_embeddings["full_attention"],
            "position_ids": state.position_ids,
            "past_key_values": cache,
            "use_cache": True
        }
        
        return layer.cls(**kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        