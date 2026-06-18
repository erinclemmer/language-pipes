import torch
from typing import Callable, Dict
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

# transformers: modeling_gemma3.py
def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask

class Gemma3Model:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig,
        mask_kwargs: Dict[str, any] # pyright: ignore[reportGeneralTypeIssues]
    ) -> LLmComputationState:
        del mask_kwargs["cache_position"]
        sliding_mask_kwargs = mask_kwargs.copy()
        if config.use_bidirectional_attention:
            mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
            sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(config.sliding_window)

        full_causal_mask: torch.Tensor = create_causal_mask(**mask_kwargs) # type: ignore
        sliding_causal_mask: torch.Tensor = create_sliding_window_causal_mask(**mask_kwargs) # type: ignore

        state.causal_mask = {
            "full_attention": full_causal_mask,
            "sliding_attention": sliding_causal_mask
        }
        
        state.position_embeddings = {
            "full_attention": AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids, "full_attention"),
            "sliding_attention": AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids, "sliding_attention")
        }
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        layer_type: str = layer.cls.config.layer_types[layer.cls.layer_idx] # type: ignore
        kwargs = { # pyright: ignore[reportUnknownVariableType]
            "hidden_states": state.state,
            "attention_mask": state.causal_mask[layer_type],
            "position_embeddings": state.position_embeddings[layer_type], # pyright: ignore[reportArgumentType]
            "position_ids": state.position_ids,
            "past_key_values": cache
        }
        
        return layer.cls(**kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        