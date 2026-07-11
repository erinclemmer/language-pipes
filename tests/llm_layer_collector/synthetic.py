"""Tiny-checkpoint builders for the Tier 1 parity suite.

Builds a random HF model from the real config class, saves it with
``save_pretrained(safe_serialization=True)`` so the on-disk format (safetensors
shards + config.json) is exactly what :class:`LlmLayerCollector` parses, then
optionally post-processes the shards:

* **fp8** — quantize projection weights to ``float8_e4m3fn`` and write a matching
  ``<name>.weight_scale_inv``. The in-memory reference model is mutated to the
  *dequantized* values first, so the roundtrip through ``dequantize_fp8_weights``
  is numerically exact (fp8 rounding is applied on both sides).
* **multimodal renaming** — rename keys to the ``model.language_model.*`` (Gemma4)
  or ``language_model.model.*`` (Mistral3) nesting that real multimodal
  checkpoints ship, exercising the collector's name-derivation logic.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

from .specs import (
    TinyModelSpec,
    KEY_STYLE_GEMMA4_MM,
    KEY_STYLE_MISTRAL3_MM,
)

# float8_e4m3fn representable maximum magnitude.
FP8_MAX = 448.0

# Decoder-layer projection weights that real fp8 checkpoints quantize.
_FP8_PROJ_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)


@dataclass
class TinyCheckpoint:
    model: torch.nn.Module          # the HF *ForCausalLM reference (post-quantization)
    config: PretrainedConfig        # unwrapped text config (what the collector sees)
    model_dir: str
    cache_file: str


def _is_fp8_proj(key: str) -> bool:
    return "layers." in key and key.endswith(_FP8_PROJ_SUFFIXES)


def _quantize_model_fp8(model: torch.nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-tensor fp8-quantize projection weights, mutating the model to the
    dequantized values in place. Returns ``{key: (fp8_weight, scale)}`` to write
    into the shards."""
    quantized: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    params = dict(model.named_parameters())
    for name, weight in params.items():
        if not _is_fp8_proj(name):
            continue
        max_abs = weight.detach().abs().max()
        scale = (max_abs / FP8_MAX) if max_abs > 0 else torch.tensor(1.0)
        q = (weight.detach() / scale).to(torch.float8_e4m3fn)
        dequant = q.to(torch.float32) * scale
        weight.data.copy_(dequant.to(weight.dtype))
        quantized[name] = (q, scale.reshape(1))
    return quantized


def _rename_key(key: str, key_style: str) -> str:
    if key_style == KEY_STYLE_GEMMA4_MM:
        if key.startswith("model."):
            return "model.language_model." + key[len("model."):]
        if key == "lm_head.weight":
            return "model.language_model.lm_head.weight"
    elif key_style == KEY_STYLE_MISTRAL3_MM:
        if key.startswith("model."):
            return "language_model.model." + key[len("model."):]
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
    return key


def _max_shard_size(model: torch.nn.Module, shards: int) -> str:
    """Pick a ``max_shard_size`` that yields at least ``shards`` files."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    per_kb = max(1, (total // max(1, shards)) // 1024)
    return f"{per_kb}KB"


def build_tiny_checkpoint(spec: TinyModelSpec, dest_dir: str, seed: int = 0) -> TinyCheckpoint:
    torch.manual_seed(seed)
    config = spec.build_config()
    # AutoModelForCausalLM.from_config resolves the correct *ForCausalLM class for
    # the config's model_type, so specs only need to name the config class.
    model = AutoModelForCausalLM.from_config(config).eval()

    quantized: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if spec.fp8:
        quantized = _quantize_model_fp8(model)

    model.save_pretrained(
        dest_dir,
        safe_serialization=True,
        max_shard_size=_max_shard_size(model, spec.shards),
    )

    needs_rewrite = spec.fp8 or spec.key_style != "standard"
    if needs_rewrite:
        _rewrite_shards(dest_dir, spec, quantized)

    cache_file = os.path.join(dest_dir, "cache.json")
    return TinyCheckpoint(
        model=model,
        config=model.config.get_text_config(),
        model_dir=dest_dir,
        cache_file=cache_file,
    )


def _rewrite_shards(
    dest_dir: str,
    spec: TinyModelSpec,
    quantized: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Rename keys and substitute fp8 weights + scales, in every shard file."""
    shard_files = [f for f in os.listdir(dest_dir) if f.endswith(".safetensors")]
    for fname in shard_files:
        path = os.path.join(dest_dir, fname)
        data = load_file(path)
        new: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            if key in quantized:
                q, scale = quantized[key]
                nk = _rename_key(key, spec.key_style)
                new[nk] = q
                new[nk[: -len(".weight")] + ".weight_scale_inv"] = scale
            else:
                new[_rename_key(key, spec.key_style)] = value
        save_file(new, path, metadata={"format": "pt"})

    # The HF-written index maps the *old* key names; the collector rebuilds its
    # cache by scanning shards (never the index), but drop it to avoid a stale
    # artifact whose keys no longer exist on disk.
    index = os.path.join(dest_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        os.remove(index)
