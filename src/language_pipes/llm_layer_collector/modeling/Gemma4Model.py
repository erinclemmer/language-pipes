import torch
from typing import Dict, Optional

from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextScaledWordEmbedding,
)

from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState


class Gemma4PerLayerEmbedder(torch.nn.Module):
    """Per-Layer Embeddings (PLE) for Gemma4.

    Wraps the three PLE weights that live only on the embedding node
    (``embed_tokens_per_layer``, ``per_layer_model_projection``,
    ``per_layer_projection_norm``) and reproduces ``Gemma4TextModel``'s
    ``get_per_layer_inputs`` + ``project_per_layer_inputs`` pipeline. The output is a
    small read-only tensor of shape ``[batch, seq, num_hidden_layers, ple_dim]`` that is
    shipped in ``JobData`` and sliced per-layer by every node.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        padding_idx = 0 if config.pad_token_id is None else config.pad_token_id

        self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_model_projection = torch.nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = config.hidden_size**-0.5

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Token-identity component (get_per_layer_inputs).
        per_layer_inputs = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        # Context-aware component (project_per_layer_inputs).
        per_layer_projection = (
            self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale


class Gemma4Model:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig,
        mask_kwargs: Dict[str, any] # pyright: ignore[reportGeneralTypeIssues]
    ) -> LLmComputationState:
        del mask_kwargs["cache_position"]

        full_causal_mask: torch.Tensor = create_causal_mask(**mask_kwargs) # type: ignore
        sliding_causal_mask: torch.Tensor = create_sliding_window_causal_mask(**mask_kwargs) # type: ignore

        state.causal_mask = {
            "full_attention": full_causal_mask,
            "sliding_attention": sliding_causal_mask,
        }

        state.position_embeddings = {
            "full_attention": AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids, "full_attention"),
            "sliding_attention": AutoRotaryEmbedding(config)(state.state.detach(), state.position_ids, "sliding_attention"),
        }
        return state

    @staticmethod
    def compute_layer(
        layer: AutoDecoderLayer,
        state: LLmComputationState,
        cache: DynamicCache
    ) -> torch.Tensor:
        layer_idx: int = layer.cls.layer_idx # type: ignore
        layer_type: str = layer.cls.config.layer_types[layer_idx] # type: ignore

        per_layer_input: Optional[torch.Tensor] = None
        if state.per_layer_inputs is not None:
            per_layer_input = state.per_layer_inputs[:, :, layer_idx, :]

        kwargs = { # pyright: ignore[reportUnknownVariableType]
            "hidden_states": state.state,
            "per_layer_input": per_layer_input,
            # Cross-node KV sharing: producer layers (last of each layer_type) write
            # full-length (k, v) here; shared layers read it. Mutated in place, then
            # written back via compute_layers / Job.set_layer so it rides JobData to the
            # next node. Always a dict (never None) — producer layers index into it even
            # when num_kv_shared_layers == 0.
            "shared_kv_states": state.shared_kv_states,
            "attention_mask": state.causal_mask[layer_type],
            "position_embeddings": state.position_embeddings[layer_type], # pyright: ignore[reportArgumentType]
            "position_ids": state.position_ids,
            "past_key_values": cache,
        }

        return layer.cls(**kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
