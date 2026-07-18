"""Model spec tables — the single touchpoint the ``llm-layer-collector`` skill edits.

Adding support for a new architecture means adding **one** :class:`TinyModelSpec`
here (copy the closest existing entry) and, optionally, one :class:`RealModelSpec`.
Everything else — checkpoint building, parity, memory-capping — is driven off these
tables by ``synthetic.py`` / ``test_tiny_models.py`` / ``test_real_models.py``.

Requires ``transformers >= 5.8`` (the tiny specs construct real HF config/model
classes, so a transformers upgrade that renames a class fails Tier 1 immediately).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from transformers.configuration_utils import PretrainedConfig

# HF config classes for every registered architecture. Imported eagerly so a
# transformers rename breaks import (a loud, early failure) rather than a test.
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config


# Multimodal weight-naming conventions. Real checkpoints for some architectures
# nest the language model under a wrapper; the synthetic builder renames tiny
# text-model keys to match so the collector's derivation logic is exercised.
KEY_STYLE_STANDARD = "standard"                 # model.layers.*, lm_head.weight
KEY_STYLE_GEMMA4_MM = "gemma4_multimodal"       # model.language_model.layers.* (Gemma4 wrapper)
KEY_STYLE_MISTRAL3_MM = "mistral3_multimodal"   # language_model.model.layers.* (Mistral3 wrapper)


@dataclass
class TinyModelSpec:
    model_type: str                       # dispatch key, e.g. "ministral3"
    config_cls: Type[PretrainedConfig]    # HF config class
    config_kwargs: Dict                   # tiny dims
    key_style: str = KEY_STYLE_STANDARD   # multimodal key nesting to emit
    fp8: bool = False                     # emit float8_e4m3fn weights + weight_scale_inv
    per_type_rope: bool = False           # gemma3/gemma4 per-layer-type masks + RoPE
    ple: bool = False                     # gemma4 per-layer-embedding ride-along
    mrope: bool = False                   # GLM-style multimodal rope (position_ids (3,1,seq))
    shards: int = 2                       # force multi-shard save to exercise iteration

    def build_config(self) -> PretrainedConfig:
        return self.config_cls(**self.config_kwargs)


# Shared tiny dims — kept small enough that a full float32 forward is < ~50 MB.
_TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)


TINY_MODEL_SPECS: List[TinyModelSpec] = [
    TinyModelSpec(
        model_type="llama",
        config_cls=LlamaConfig,
        config_kwargs=dict(_TINY),
    ),
    TinyModelSpec(
        model_type="qwen3",
        config_cls=Qwen3Config,
        config_kwargs=dict(_TINY),
    ),
    TinyModelSpec(
        model_type="phi3",
        config_cls=Phi3Config,
        # Phi3 has no head_dim arg and its default pad_token_id (32000) exceeds the
        # tiny vocab, which would trip nn.Embedding(padding_idx=...).
        config_kwargs={
            k: v for k, v in _TINY.items() if k != "head_dim"
        } | dict(pad_token_id=None, bos_token_id=1, eos_token_id=2),
    ),
    TinyModelSpec(
        model_type="qwen3_moe",
        config_cls=Qwen3MoeConfig,
        config_kwargs=dict(_TINY) | dict(
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,   # every layer is an MoE layer
            norm_topk_prob=True,
        ),
    ),
    TinyModelSpec(
        model_type="gemma3_text",
        config_cls=Gemma3TextConfig,
        # 6 layers so both attention types appear; sliding_window_pattern + the two
        # rope base freqs are what make the per-layer-type rope config well-formed.
        config_kwargs=dict(_TINY) | dict(
            num_hidden_layers=6,
            sliding_window=16,
            sliding_window_pattern=3,
            rope_theta=1_000_000,
            rope_local_base_freq=10_000,
        ),
        per_type_rope=True,
    ),
    TinyModelSpec(
        model_type="gemma4_text",
        config_cls=Gemma4TextConfig,
        config_kwargs=dict(_TINY) | dict(
            num_hidden_layers=5,
            num_key_value_heads=1,
            sliding_window=16,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=128,
        ),
        key_style=KEY_STYLE_GEMMA4_MM,
        per_type_rope=True,
        ple=True,
    ),
    TinyModelSpec(
        # Gemma4Unified is a separate architecture from Gemma4 (gemma-4-12B and up),
        # not a variant: same per-type RoPE/masks and `model.language_model.*` key
        # nesting, but no Per-Layer Embeddings.
        model_type="gemma4_unified_text",
        config_cls=Gemma4UnifiedTextConfig,
        config_kwargs=dict(_TINY) | dict(
            num_hidden_layers=5,
            num_key_value_heads=1,
            sliding_window=16,
            # Real gemma-4-12B sets these: on full_attention layers only, k/v share a
            # projection and the head geometry switches to global_head_dim +
            # num_global_key_value_heads. Leaving them at the config defaults would
            # give every layer uniform shapes and silently skip that path.
            attention_k_eq_v=True,
            global_head_dim=32,
            num_global_key_value_heads=1,
        ),
        key_style=KEY_STYLE_GEMMA4_MM,
        per_type_rope=True,
    ),
    TinyModelSpec(
        model_type="ministral3",
        config_cls=Ministral3Config,
        config_kwargs=dict(_TINY) | dict(sliding_window=None),
        key_style=KEY_STYLE_MISTRAL3_MM,
        fp8=True,
    ),
]


@dataclass
class RealModelSpec:
    """A real checkpoint for opt-in Tier 2 smoke tests.

    Expected numeric values are derived at runtime from ``config.json`` and the
    safetensors index — never hardcoded here. ``question`` / ``expected_next`` drive
    the cheap greedy-coherence assertion.

    Prompting goes through the model's own chat template when it has one — instruct
    models give far more stable greedy output when prompted the way they were tuned,
    and a raw completion prompt makes the first token an arbitrary artifact of the
    checkpoint (see skill §9). Base models with no template fall back to ``prompt``.
    """

    model_type: str
    model_id: str
    question: str = "What is the capital of France?"   # chat-template user message
    prompt: str = "The capital of France is"           # raw fallback, base models only
    expected_next: Optional[str] = None    # substring the greedy continuation must contain
    coherence_tokens: int = 8
    # Extra apply_chat_template kwargs, e.g. enable_thinking=False for Qwen3.
    template_kwargs: Dict = field(default_factory=dict)
    # Some GLM-4.1V checkpoints ship rope_scaling: null and crash at layer time;
    # inject a default before use (see skill §6).
    inject_rope_scaling: bool = False


REAL_MODEL_SPECS: List[RealModelSpec] = [
    RealModelSpec("qwen3", "Qwen/Qwen3-0.6B", expected_next="Paris",
                  template_kwargs={"enable_thinking": False}),
    # RealModelSpec("qwen3_moe", "Qwen/Qwen3-30B-A3B-Thinking-2507", expected_next="Paris"),
    RealModelSpec("llama", "meta-llama/Llama-3.2-1B", expected_next="Paris"),
    RealModelSpec("phi3", "microsoft/Phi-4-mini-instruct", expected_next="Paris"),
    RealModelSpec("gemma3_text", "google/gemma-3-1b-it", expected_next="Paris"),
    RealModelSpec("gemma4_text", "google/gemma-4-E2B-it", expected_next="Paris"),
    RealModelSpec("gemma4_unified_text", "google/gemma-4-12B-it", expected_next="Paris"),
    RealModelSpec("ministral3", "mistralai/Ministral-3-3B-Instruct-2512", expected_next="Paris"),
]
