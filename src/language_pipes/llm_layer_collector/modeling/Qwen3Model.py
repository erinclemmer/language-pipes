import torch
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

class Qwen3Model:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig
    ) -> LLmComputationState:
        state.position_embeddings = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        kwargs = {
            "past_key_values": cache,
            "hidden_states": state.state,
            "position_ids": state.position_ids,
            "use_cache": layer.config.use_cache,
            "cache_position": state.cache_position,
            "position_embeddings": state.position_embeddings,
            "attention_mask": state.causal_mask["full_attention"],
        }
        
        return layer.cls(**kwargs)
        