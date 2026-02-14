import copy
import torch
from typing import Dict
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_sliding_window_causal_mask
from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

class Gemma3Model:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig,
        mask_kwargs: Dict[str, any] # pyright: ignore[reportGeneralTypeIssues]
    ) -> LLmComputationState:
        state.position_embeddings_global = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
        c: PretrainedConfig = copy.deepcopy(config)
        c.rope_theta = config.rope_local_base_freq
        c.rope_scaling = { "rope_type": "default" }
        state.position_embeddings_local = AutoRotaryEmbedding(c)(state.state.detach(), state.position_ids)
        state.causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        kwargs = { # pyright: ignore[reportUnknownVariableType]
            "hidden_states": state.state,
            "position_embeddings_global": state.position_embeddings_global,
            "position_embeddings_local": state.position_embeddings_local,
            "attention_mask": state.causal_mask[layer.cls.attention_type], # pyright: ignore[reportArgumentType]
            "position_ids": state.position_ids,
            "past_key_values": cache,
            "use_cache": True,
            "cache_position": state.cache_position
        }
        
        return layer.cls(**kwargs)[0] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        