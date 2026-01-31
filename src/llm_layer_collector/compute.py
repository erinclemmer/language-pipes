import copy
import torch
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import DynamicCache

from llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding
from llm_layer_collector.state_obj import LLmComputationState

def compute_embedding(
        input_embedder: torch.nn.Embedding,
        input_ids: torch.Tensor,
        config: PretrainedConfig,
        cache: DynamicCache,
        chunked_prefill: bool = False
    ) -> LLmComputationState:
    """
    Compute embeddings and prepare computation state for transformer layers.
    
    Args:
        input_embedder: The embedding layer
        input_ids: Token IDs to embed
        config: Model configuration
        cache: KV cache (DynamicCache)
        chunked_prefill: If True, process all input_ids even when cache is non-empty.
                        Use this for chunked prefill where multiple tokens are processed
                        at once after the first chunk.
    """
    device = input_embedder.weight.device

    state = LLmComputationState()
    
    past_seen_tokens = cache.get_seq_length()
    
    # Determine which tokens to embed:
    # - First prefill (cache empty): embed all input_ids
    # - Chunked prefill (cache non-empty, chunked_prefill=True): embed all input_ids
    # - Decode phase (cache non-empty, chunked_prefill=False): embed only last token
    if past_seen_tokens == 0 or chunked_prefill:
        input_seq = input_ids
    else:
        input_seq = torch.tensor([[input_ids[:, -1]]]).to(device)
    
    state.state = input_embedder(input_seq.to(device))

    converter = AttentionMaskConverter(is_causal=True)
    L = input_seq.size()[1]

    attention_mask = converter.to_causal_4d(
        batch_size=1,
        query_length=L,
        key_value_length=past_seen_tokens + L,
        dtype=state.state.dtype,
        device=device
    )
    
    state.cache_position = torch.arange(
        past_seen_tokens, end=past_seen_tokens + L, device=device
    )
    
    state.position_ids = state.cache_position.unsqueeze(0)

    mask_kwargs = {
        "config": config,
        "input_embeds": state.state.detach(),
        "attention_mask": attention_mask,
        "cache_position": state.cache_position,
        "past_key_values": cache,
        "position_ids": state.position_ids
    }
    
    state.causal_mask["full_attention"] = create_causal_mask(**mask_kwargs)
    state.causal_mask["sliding_attention"] = None

    try:
        if "sliding_attention" in config.layer_types or config.sliding_window is not None:
            state.causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    except AttributeError:
        pass

    if config.model_type == 'gemma3_text':
        state.position_embeddings_global = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
        configCopy = copy.deepcopy(config)
        configCopy.rope_theta = configCopy.rope_local_base_freq
        configCopy.rope_scaling = {"rope_type": "default"}
        
        state.position_embeddings_local = AutoRotaryEmbedding(configCopy)(state.state.detach(), state.position_ids)
    else:
        state.position_embeddings = AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids)
    
    return state

def compute_head(
        head: torch.nn.Linear,
        state: torch.Tensor,
        topk: int = 1
    ) -> torch.Tensor:
    state = head(state[:, -1, :])
    probs = torch.softmax(state, dim=-1)
    return torch.topk(probs, topk).indices
