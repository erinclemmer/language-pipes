import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.modeling.Qwen3Model import Qwen3Model
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

class StaticAutoModel:
    @staticmethod
    def compute_embedding(
        input_embedder: torch.nn.Embedding,
        input_ids: torch.Tensor,
        config: PretrainedConfig,
        cache: DynamicCache
    ) -> LLmComputationState:
        # TODO Automatically trim state etc depending upon the state of the cache
        device = input_embedder.weight.device

        state = LLmComputationState()
        
        past_seen_tokens = cache.get_seq_length()
        
        state.state = input_embedder(input_ids.to(device))

        converter = AttentionMaskConverter(is_causal=True)
        L = input_ids.size()[1]

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

        match config.model_type:
            case "qwen3":
                Qwen3Model.compute_embedding(state, config)
                
            case "qwen3_moe":
                Qwen3Model.compute_embedding(state, config)

        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        match layer.config.model_type:
            case "qwen3":
                return Qwen3Model.compute_layer(layer, state, cache)
            case "qwen3_moe":
                return Qwen3Model.compute_layer(layer, state, cache)

    @staticmethod
    def compute_head(
        head: torch.nn.Linear,
        input_ids: torch.Tensor,
        device: str,
        state: torch.Tensor,
        top_k: int = 1,
        top_p: float = 1,
        min_p: float = 0,
        temperature: float = 1,
        presence_penalty: float = 0
    ) -> torch.Tensor:
        with torch.inference_mode():
            state_on_device = state.detach().clone()[:, -1, :].to(device)
            logits = torch.nn.functional.linear(
                state_on_device, 
                head.weight, 
                head.bias
            ).flatten()
            del state_on_device
            
            # Apply presence penalty to discourage token repetition
            # Subtracts penalty from logits of tokens that have already appeared
            if presence_penalty != 0 and len(input_ids) > 0:
                unique_tokens = set(input_ids)
                for token_id in unique_tokens:
                    logits[token_id] -= presence_penalty

            if temperature == 0:
                # Greedy decoding - just pick the top token
                head = int(logits.argmax().item())
            else:
                # Apply temperature scaling to logits before softmax
                # Lower temperature = sharper distribution (more deterministic)
                # Higher temperature = flatter distribution (more random)
                scaled_logits = logits / temperature

                # Apply min_p filtering if specified (min_p > 0)
                # Remove tokens with probability < min_p * max_probability
                if min_p > 0:
                    probs = torch.nn.functional.softmax(scaled_logits, dim=0)
                    max_prob = probs.max()
                    min_prob_threshold = min_p * max_prob
                    indices_to_remove = probs < min_prob_threshold
                    scaled_logits[indices_to_remove] = float('-inf')

                # Apply top_p (nucleus) filtering if specified (top_p < 1.0)
                # Mask out tokens outside the top-p cumulative probability mass
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=0)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    # Find indices to remove (cumulative prob exceeds top_p)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift to keep at least one token
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    # Scatter -inf back to original positions
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = float('-inf')

                # Apply top_k filtering if specified (top_k > 0)
                # Mask out non-top-k tokens by setting them to -inf
                if top_k > 0:
                    k = min(top_k, scaled_logits.size(0))
                    top_k_values, _ = torch.topk(scaled_logits, k)
                    threshold = top_k_values[-1]
                    scaled_logits = torch.where(scaled_logits < threshold, torch.tensor(float('-inf'), device=scaled_logits.device), scaled_logits)

                probabilities = torch.nn.functional.softmax(scaled_logits, dim=0)
                head = int(torch.multinomial(probabilities, num_samples=1).item())
            
            del logits
            return head