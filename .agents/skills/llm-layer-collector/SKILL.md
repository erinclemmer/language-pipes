---
name: llm-layer-collector
description: Guide for adding HuggingFace transformer model architecture support to the llm_layer_collector layer-loading and compute dispatch system.
---

# Skill: llm_layer_collector — Adding New Model Support

## Overview

The `llm_layer_collector` is the abstraction layer that allows Language Pipes to work with different HuggingFace transformer model architectures at the individual layer level. It loads model components (embedding, norm, head, decoder layers) from safetensor shards and dispatches computation to model-specific implementations based on the `model_type` field in the model's HuggingFace `config.json`.

To add support for a new model architecture you must modify **4 existing files** and create **1 new file**. All changes are within `src/language_pipes/llm_layer_collector/`.

---

## Architecture Summary

```
llm_layer_collector/
├── layer_collector.py          # Core class: loads shards, caches layer→file mappings
├── load_layer.py               # Loads decoder layer weights from safetensors into AutoDecoderLayer
├── cache.py                    # Builds/reads the shard metadata cache
├── helpers.py                  # Shared utilities (config loading, tensor loading)
├── state_obj.py                # LLmComputationState — carries hidden state through the pipeline
├── auto/
│   ├── auto_layer.py           # Maps model_type → DecoderLayer class from transformers
│   ├── auto_rms.py             # Maps model_type → RMSNorm class from transformers
│   ├── auto_rotary.py          # Maps model_type → RotaryEmbedding class from transformers
│   └── static_auto_model.py    # Dispatches compute_embedding / compute_layer / compute_head by model_type
└── modeling/
    ├── LlamaModel.py           # Bespoke implementation for Llama-family models
    └── Qwen3Model.py           # Bespoke implementation for Qwen3/Qwen3-MoE models
```

### How computation flows

1. `StaticAutoModel.compute_embedding()` creates a `LLmComputationState` with the input embedding, causal mask, cache position, and position IDs. It then dispatches to the model-specific `compute_embedding()` to set up position embeddings (e.g., rotary embeddings).
2. `StaticAutoModel.compute_layer()` dispatches to the model-specific `compute_layer()` which calls the underlying transformers `DecoderLayer` with the correct keyword arguments for that architecture.
3. `StaticAutoModel.compute_head()` is model-agnostic — it runs the final linear projection and sampling.

### The `model_type` key

Every HuggingFace model has a `config.json` with a `"model_type"` field (e.g. `"llama"`, `"qwen3"`, `"gemma3_text"`, `"qwen3_moe"`). This string is the dispatch key used everywhere in the auto classes. You can find it by looking at the model's `config.json` on HuggingFace or by running:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("org/model-name")
print(config.model_type)
```

---

## Step-by-Step: Adding a New Model

Throughout these steps, replace `NewModel` with your model's name (e.g. `Gemma3Model`, `MistralModel`) and `"new_model_type"` with the exact `model_type` string from the HuggingFace config.

### Step 1: Register the Decoder Layer in `auto/auto_layer.py`

Import the decoder layer class from the `transformers` library and add it to the `mapper` dict.

```python
# Add import
from transformers.models.<module_name>.modeling_<module_name> import NewModelDecoderLayer

# Add to mapper dict
mapper = {
    "llama": LlamaDecoderLayer,
    "qwen3": Qwen3DecoderLayer,
    "gemma3_text": Gemma3DecoderLayer,
    "qwen3_moe": Qwen3MoeDecoderLayer,
    "new_model_type": NewModelDecoderLayer,  # ← ADD THIS
}
```

**How to find the correct import:** Search the transformers source for `class <Name>DecoderLayer`. The module path follows the pattern `transformers.models.<name>.modeling_<name>`. You can find the transformers source code in env/lib/python3.14/site-packages/transformers

### Step 2: Register the RMS Norm in `auto/auto_rms.py`

Import the RMS norm class and add it to the `mapper` dict.

```python
# Add import
from transformers.models.<module_name>.modeling_<module_name> import NewModelRMSNorm

# Add to mapper dict
mapper = {
    "llama": LlamaRMSNorm,
    "qwen3": Qwen3RMSNorm,
    "gemma3_text": Gemma3RMSNorm,
    "qwen3_moe": Qwen3MoeRMSNorm,
    "new_model_type": NewModelRMSNorm,  # ← ADD THIS
}
```

**Note:** Some models may use `LayerNorm` instead of `RMSNorm`. The `AutoRMSNorm` constructor expects `(hidden_size, eps=rms_norm_eps)` — verify the new norm class uses a compatible signature, or adjust accordingly.

### Step 3: Register the Rotary Embedding in `auto/auto_rotary.py`

Import the rotary embedding class and add it to the `mapper` dict.

```python
# Add import
from transformers.models.<module_name>.modeling_<module_name> import NewModelRotaryEmbedding

# Add to mapper dict
mapper = {
    "llama": LlamaRotaryEmbedding,
    "qwen3": Qwen3RotaryEmbedding,
    "gemma3_text": Gemma3RotaryEmbedding,
    "qwen3_moe": Qwen3MoeRotaryEmbedding,
    "new_model_type": NewModelRotaryEmbedding,  # ← ADD THIS
}
```

**Note:** Not all architectures use rotary embeddings. If the model uses a different positional encoding scheme, you may need to handle this differently in the modeling file (Step 5).

### Step 4: Add Dispatch Cases in `auto/static_auto_model.py`

Import your new model class and add `case` branches to the `match` statements in both `compute_embedding()` and `compute_layer()`.

```python
# Add import at top
from language_pipes.llm_layer_collector.modeling.NewModel import NewModel

# In compute_embedding(), add a case:
match config.model_type:
    case "qwen3":
        Qwen3Model.compute_embedding(state, config)
    case "qwen3_moe":
        Qwen3Model.compute_embedding(state, config)
    case "llama":
        LlamaModel.compute_embedding(state, config)
    case "new_model_type":                          # ← ADD THIS
        NewModel.compute_embedding(state, config)   # ← ADD THIS

# In compute_layer(), add a case:
match layer.config.model_type:
    case "qwen3":
        return Qwen3Model.compute_layer(layer, state, cache)
    case "qwen3_moe":
        return Qwen3Model.compute_layer(layer, state, cache)
    case "llama":
        return LlamaModel.compute_layer(layer, state, cache)
    case "new_model_type":                                      # ← ADD THIS
        return NewModel.compute_layer(layer, state, cache)      # ← ADD THIS
```

### Step 5: Create the Bespoke Model File `modeling/NewModel.py`

Create a new file at `src/language_pipes/llm_layer_collector/modeling/NewModel.py` with two static methods.

```python
import torch
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from language_pipes.llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding

from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

class NewModel:
    @staticmethod
    def compute_embedding(
        state: LLmComputationState,
        config: PretrainedConfig
    ) -> LLmComputationState:
        # Most models use standard rotary embeddings:
        state.position_embeddings = AutoRotaryEmbedding(config)(
            state.state.detach(), state.position_ids
        )
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
            "use_cache": True,
            "cache_position": state.cache_position,
            "position_embeddings": state.position_embeddings,
            "attention_mask": state.causal_mask["full_attention"],
        }
        
        return layer.cls(**kwargs)
```

**Customization guidance for `compute_layer`:**

- The `kwargs` dict must match what the transformer's `DecoderLayer.forward()` method expects. Check the transformers source for the specific model.
- Most standard decoder-only transformers (Llama, Qwen3, Mistral, etc.) use the same kwargs shown above.
- Some architectures may need additional kwargs (e.g., sliding window attention masks via `state.causal_mask["sliding_attention"]`, separate local/global position embeddings, etc.).
- Look at the `forward()` signature of the model's `DecoderLayer` class in the transformers library to determine the exact parameters needed.

**Customization guidance for `compute_embedding`:**

- Most models use `AutoRotaryEmbedding` which dispatches to the correct rotary implementation.
- Some models may need additional position embedding setup (e.g., separate local and global embeddings). The `LLmComputationState` has `position_embeddings_local` and `position_embeddings_global` fields available for this.
- Per-type RoPE (Gemma3/Gemma4): `AutoRotaryEmbedding(config)(x, position_ids, layer_type)` — the third arg is required for these models, and `compute_layer` must key both the mask and the RoPE by `config.layer_types[layer_idx]` (`"full_attention"` / `"sliding_attention"`). The mask dict is built once in `compute_embedding` via `create_causal_mask` + `create_sliding_window_causal_mask` (delete `cache_position` from `mask_kwargs` first — these models don't pass it).

---

## Architectures Needing Extra Per-Forward Tensors (Ride-Along State)

Most models only need `state`, masks, RoPE, and the per-node KV cache. Some newer architectures need an **extra tensor computed once at embed time that every layer node must see** (example: Gemma4 Per-Layer Embeddings / PLE). The pipeline already ships such tensors via `JobData` — masks and RoPE ride along exactly this way. To add a new ride-along field:

1. **`state_obj.py`** — add an `Optional[Tensor] = None` field to `LLmComputationState`.
2. **`jobs/job_data.py`** — add the matching field to `JobData`, then serialize it in `to_bytes`/`from_bytes` (reuse the single-tensor presence-flag pattern: `write_int(0)` when `None`, else `write_int(1)` + `write_bytes(tensor_to_bytes(...))`), and move/detach it in `computationStateToJobData`, `jobDataToComputationState`, and `detachCompState`.
3. **`static_auto_model.compute_embedding`** — take an optional loader/module param (default `None`, so every other model is unaffected) and populate the state field for the relevant `model_type`.
4. **`modeling/end_model.py`** — load the weights and pass them into `compute_embed`; null them in `clean_up`.
5. **`modeling/NewModel.compute_layer`** — read the field off `state` (slice per-layer if needed: `state.per_layer_inputs[:, :, layer_idx, :]`).

**Memory locality is the critical constraint.** If the extra weights are huge (Gemma4's `embed_tokens_per_layer` is the single largest tensor in the checkpoint), load them **only on the head/embedding node** (`EndModel`), never on `LlmModel` layer nodes. The loader returns `None` for models that don't use the feature, so layer nodes never touch it. The ride-along output is small and read-only — that's what makes shipping it cheap.

If the field is **mutated as it flows** (rare — e.g. cross-node KV sharing), you also need write-back: extend `compute_layers` to return the mutated value and `Job.set_layer` to write it into `job.data`, so it serializes to the next node. A read-only ride-along (like PLE) needs none of that.

---

## Verifying Your Changes

### 1. Check that the model_type is recognized

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("org/model-name")
print(config.model_type)  # Should match your key in all mapper dicts
```

### 2. Run the layer collector test suite

The suite lives in `tests/llm_layer_collector/` and has three tiers. **The only
file you edit to add a model is `specs.py`** — everything else (tiny-checkpoint
building, HF parity, memory-capping) is driven off the spec tables.

**Add one `TinyModelSpec` to `specs.py`.** Copy the closest existing architecture
and set the tiny dims + flags (`per_type_rope`, `ple`, `fp8`, `key_style`,
`mrope`). Then run the tiny-parity suite — no download, KBs of memory, tight
float32 tolerances:

```bash
python -m unittest tests.llm_layer_collector.test_units tests.llm_layer_collector.test_tiny_models
python -m unittest tests.llm_layer_collector.test_tiny_models -k <model_type>
```

`test_tiny_models` builds a random tiny model from your config class, saves it as
real safetensors shards, loads it back through `LlmLayerCollector`, and compares
the full forward path (single-shot, chunked-prefill, and decode) against the HF
reference model's own `last_hidden_state` (`cosine ≥ 0.9999`, `max_abs_diff < 1e-4`).
This is the primary CI gate — it fails loudly on the exact wiring bugs (unscaled
fp8, wrong lm_head, missing embed scaling, uninitialized MoE experts) that the
Debugging Playbook below exists for. Architecture quirks become spec-driven
assertions (per-type masks/RoPE, PLE ride-along shape, fp8 dequant equality).

If a tiny spec surfaces a gap, **fix the spec table, not the test factory** — the
factory is architecture-agnostic. Common tiny-config gotchas: Phi3's default
`pad_token_id` exceeds a tiny vocab (set `pad_token_id=None`); Gemma3 per-type
RoPE needs `sliding_window_pattern` + `rope_local_base_freq`; Gemma4/Ministral3
real checkpoints nest weights under a multimodal wrapper, so the spec's
`key_style` renames tiny keys to match (`gemma4_multimodal` = `model.language_model.*`,
`mistral3_multimodal` = `language_model.model.*`).

**Only then, optionally, add a `RealModelSpec`** for an opt-in real-checkpoint
smoke test (prefer the smallest real checkpoint). Expected values are derived
from `config.json` / the safetensors index — never hardcoded. Tier 2 is gated so
default CI never downloads, runs each model in a memory-capped subprocess (10 GB
RSS budget in `memguard.py`), and streams the layer stack one window at a time via
`run_stack_windowed` so even a 16 GB-resident model fits under the cap:

```bash
LP_RUN_MODEL_TESTS=1       python -m unittest tests.llm_layer_collector.test_real_models -k <model_type>
LP_RUN_MODEL_TESTS=nightly python -m unittest tests.llm_layer_collector.test_real_models   # + short decode
```

### 3. Validate a sub-component by hand (ad-hoc, no checkpoint)

Tier 1 above *is* the automated synthetic-parity gate. For one-off debugging of a
single component you can still build a tiny random model from the HF config class,
copy its weights into your modeling path, and compare hidden states directly —
stronger evidence than an end-to-end run because it isolates *your* dispatch code
from tokenizer/sampling noise.

```bash
PYTHONPATH=src python - <<'PY'
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from language_pipes.llm_layer_collector.auto.static_auto_model import StaticAutoModel
from language_pipes.llm_layer_collector.modeling.Gemma4Model import Gemma4Model

cfg = Gemma4TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    head_dim=16, hidden_size_per_layer_input=8, vocab_size_per_layer_input=128)
cfg._attn_implementation = "sdpa"
hf = Gemma4TextModel(cfg).eval()
input_ids = torch.randint(0, 128, (1, 6))
with torch.no_grad():
    ref = hf(input_ids=input_ids, use_cache=False).last_hidden_state

class W:  # wrap an HF decoder layer to look like AutoDecoderLayer (needs .cls)
    def __init__(self, l): self.cls = l
cache = DynamicCache(config=cfg)
state = StaticAutoModel.compute_embedding(input_ids.shape[1], 64, hf.embed_tokens,
    input_ids, cfg, cache)            # add per_layer_embedder=... for PLE models
with torch.inference_mode():
    for l in hf.layers:
        state.state = Gemma4Model.compute_layer(W(l), state, cache)
ours = hf.norm(state.state)
print("cosine_last:", torch.nn.functional.cosine_similarity(ref[:, -1, :], ours[:, -1, :]).item())
print("max_abs_diff:", (ref - ours).abs().max().item())  # want ~1.0 and ~0.0
PY
```

The same trick validates a sub-component in isolation (e.g. copy the 3 PLE weights into `Gemma4PerLayerEmbedder` and compare its output to HF's `get_per_layer_inputs` + `project_per_layer_inputs`).

---

## Debugging Playbook (When Test Passes but Output Is Garbled)

Sometimes shape/assertion tests pass but generated text is nonsense. This usually means model wiring is *nearly* correct, but one architecture-specific behavior is missing.

### 1) Compare against HuggingFace reference model

If generation is garbled, compare your forward path against `AutoModelForCausalLM` for the same prompt.

Useful command:

```bash
PYTHONPATH=/home/erin/prog/language-pipes/src python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from language_pipes.llm_layer_collector.layer_collector import LlmLayerCollector
from language_pipes.llm_layer_collector import StaticAutoModel

model_dir='/home/erin/.cache/language_pipes/models/google/gemma-3-1b-it/data'
cache_file='/home/erin/.cache/language_pipes/models/google/gemma-3-1b-it/cache.json'

tok=AutoTokenizer.from_pretrained(model_dir)
hf=AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).eval()
ids=tok.apply_chat_template([
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"How do molecules work?"}
], tokenize=True, add_generation_prompt=True, return_tensors='pt')

with torch.no_grad():
    out=hf.model(input_ids=ids, use_cache=True)
    hf_last=out.last_hidden_state[:, -1, :].float()

collector=LlmLayerCollector(model_dir, cache_file)
emb=collector.load_input_embedding(); norm=collector.load_norm(); layers=collector.load_layer_set(0, collector.config.num_hidden_layers - 1)
cache=DynamicCache()
state=StaticAutoModel.compute_embedding(ids.shape[1], 32, emb, ids, collector.config, cache)
for lyr in layers:
    state.state=StaticAutoModel.compute_layer(lyr, state, cache)
ours_last=norm(state.state)[:, -1, :].float()

print('cosine_similarity_last_hidden =', torch.nn.functional.cosine_similarity(hf_last, ours_last).item())
PY
```

If cosine similarity is very low (e.g. ~0.07), divergence is happening before sampling, usually in embedding/mask/positional wiring.

### 2) Inspect HF model internals directly

Use `inspect.getsource(...)` on the exact architecture classes to verify required behavior:

```bash
python - <<'PY'
import inspect
from transformers.models.gemma3 import modeling_gemma3 as m
print(inspect.getsource(m.Gemma3TextModel.__init__))
print(inspect.getsource(m.Gemma3DecoderLayer.forward))
PY
```

This quickly reveals architecture-specific details (extra kwargs, local/global RoPE, scaled embeddings, etc.).

### 3) Verify layer-state loading is complete

Before blaming sampling, confirm no silent state_dict mismatch:

```bash
PYTHONPATH=/home/erin/prog/language-pipes/src python - <<'PY'
import torch, json
from transformers import AutoConfig
from language_pipes.llm_layer_collector.load_layer import get_shard_data
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer

model_dir='/home/erin/.cache/language_pipes/models/google/gemma-3-1b-it/data'
cache_file='/home/erin/.cache/language_pipes/models/google/gemma-3-1b-it/cache.json'
with open(cache_file) as f: layer_files=json.load(f)
cfg=AutoConfig.from_pretrained(model_dir)
shard_data=get_shard_data(0,0,torch.device('cpu'),model_dir,'model.layers.',layer_files,torch.float16)

torch.set_default_device('meta'); lyr=AutoDecoderLayer(cfg,0)
torch.set_default_device('cpu'); lyr=lyr.to_empty(device=torch.device('cpu'))
prefix='model.layers.0.'
state={k[len(prefix):]:v for k,v in shard_data.items() if k.startswith(prefix)}
res=lyr.cls.load_state_dict(state, strict=False)
print('missing_keys=', len(res.missing_keys), 'unexpected_keys=', len(res.unexpected_keys))
PY
```

### 4) NaNs appearing partway through the layer stack (dtype instability)

If `compute_layers` runs cleanly through the first several layers and then produces `NaN`/`Inf` partway through the stack, the most likely cause is **running the model in `float16`** rather than `bfloat16` or `float32`.

Symptoms:

- Forward pass starts fine, hidden-state magnitudes grow each layer, then one layer overflows and downstream layers produce all-NaN tensors.
- Output is garbage tokens or `argmax` of NaN logits.
- The failing layer index is roughly consistent across runs but shifts with prompt length / batch.

Why this happens: newer architectures (notably **Gemma 3**, but also some Qwen/Phi variants) have activation magnitudes that exceed `float16`'s ~65k range. Residual additions accumulate across layers, eventually overflowing. `bfloat16` has the same 8-bit exponent as `float32`, so it tolerates the same dynamic range — no overflow.

Fix: load layers in `torch.bfloat16` (or `torch.float32` if hardware lacks fast bf16). Check every `to(dtype=...)` and `get_shard_data(..., dtype=...)` call in the load path. A single sub-module left in fp16 (e.g. RMSNorm weights) is enough to reintroduce the overflow.

Quick localizer to drop into `compute.py` during debugging:

```python
for i, lyr in enumerate(layers[start_layer:]):
    comp_state.state = StaticAutoModel.compute_layer(lyr, comp_state, cache).detach()
    if not torch.isfinite(comp_state.state).all():
        attn_type = getattr(lyr.cls, "attention_type", "n/a")
        max_abs = comp_state.state.abs().max().item()
        raise RuntimeError(
            f"NaN/Inf at layer {start_layer + i} "
            f"(attn_type={attn_type}, dtype={comp_state.state.dtype}, max_abs={max_abs})"
        )
```

If the failing layer is always a `sliding_attention` layer and dtype is already bf16/fp32, suspect the sliding-window mask (e.g. `state.causal_mask["sliding_attention"]` is `None` when it shouldn't be) before suspecting overflow.

Hardware notes for `bfloat16`:

- NVIDIA **Ampere (RTX 30xx, A100) and newer** — native tensor-core bf16, fast.
- NVIDIA **Turing (RTX 20xx, T4) and older** — no native bf16 path; PyTorch emulates and it may be slower than fp16. Correctness is still fine. If too slow, fall back to `float32`.
- Apple Silicon (MPS), AVX-512-BF16 CPUs (Sapphire Rapids, Zen 4+), and AMD MI200+ all support bf16 natively.
- Check at runtime with `torch.cuda.is_bf16_supported()`.

### 5) Gemma3-specific gotchas (learned)

- **Input embedding must be scaled** using `Gemma3TextScaledWordEmbedding` (`embed_scale=sqrt(hidden_size)`), not plain `torch.nn.Embedding`.
- **LM head should be bias-free** (`bias=False`) for this decoder-only setup.
- Matching mask APIs alone is not enough if embedding behavior differs from HF architecture.

### 6) GLM-4.1V-specific gotchas (learned)

- Some GLM-4.1V checkpoints (e.g. `zai-org/GLM-4.1V-9B-Thinking`) may ship `config.json` with `"rope_scaling": null`.
- In current `transformers`, `Glm4vTextAttention.forward()` unconditionally accesses:

  ```python
  self.rope_scaling["mrope_section"]
  ```

  so a `None` value crashes at runtime during layer execution with:
  `TypeError: 'NoneType' object is not subscriptable`.
- Practical fix in Language Pipes: after loading `Glm4vTextConfig`, inject a default when missing:

  ```python
  if config.rope_scaling is None:
      config.rope_scaling = {
          "rope_type": "default",
          "mrope_section": [16, 24, 24],
      }
  ```

  For GLM-4.1V-9B, this split is compatible with `head_dim=128` because
  `sum([16,24,24]) * 2 == 128`, matching how `apply_multimodal_rotary_pos_emb()` uses `mrope_section`.

### 7) Gemma4-specific gotchas (learned)

- **`model_type` is `"gemma4_text"`**, module `transformers.models.gemma4.modeling_gemma4`. Classes: `Gemma4TextDecoderLayer`, `Gemma4RMSNorm`, `Gemma4TextRotaryEmbedding`, `Gemma4TextScaledWordEmbedding`. Like Gemma3, it needs the scaled word embedding (`embed_scale=sqrt(hidden_size)`) and per-type masks/RoPE keyed by `config.layer_types[layer_idx]`.
- **Per-Layer Embeddings (PLE)** — active when `config.hidden_size_per_layer_input` is truthy. `compute_layer` **must** pass `per_layer_input` (the layer's slice of a `[batch, seq, num_layers, ple_dim]` tensor) or the forward silently goes wrong / errors. This is a ride-along tensor (see "Architectures Needing Extra Per-Forward Tensors"): computed once on the head node from the three weights `model.embed_tokens_per_layer.weight`, `model.per_layer_model_projection.weight`, `model.per_layer_projection_norm.weight`, reproducing `Gemma4TextModel.get_per_layer_inputs` + `project_per_layer_inputs`. **Never load `embed_tokens_per_layer` on layer nodes** — it's the largest tensor in the checkpoint.
- **`shared_kv_states` must be a `dict`, not `None`** — `Gemma4TextDecoderLayer.forward` accepts `shared_kv_states=None`, but even with `num_kv_shared_layers == 0` the *producer* layers (`store_full_length_kv`, i.e. the last layer of each `layer_type`) execute `shared_kv_states[self.layer_type] = ...`. Passing `None` raises `TypeError: 'NoneType' object is not subscriptable`. Pass `{}` if you aren't implementing cross-node KV sharing — the writes are harmless because no layer reads them within a single forward when nothing is a `is_kv_shared_layer`.
- **Cross-node KV sharing** (when `num_kv_shared_layers > 0`) is the one case that needs *mutated* ride-along state with write-back through `compute_layers`/`Job.set_layer`. The dict is keyed by `layer_type` (~2 entries), and shared layers legitimately lack `k_proj/v_proj/k_norm/v_norm` — `load_layers` already uses `load_state_dict(strict=False)`, so no load-path change is needed.
- **MoE / double-wide MLP** (`enable_moe_block`, `Gemma4TextRouter`/`Gemma4TextExperts`) is internal to `Gemma4TextDecoderLayer` — no special handling in the modeling file.
- **bf16 only** (same as Gemma3 — fp16 overflows).

### 8) Ministral3 / FP8-quantized checkpoint gotchas (learned)

- **`model_type` is `"ministral3"`** (text_config inside a top-level `"mistral3"` multimodal config), module `transformers.models.ministral3.modeling_ministral3`. The decoder layer takes the standard Llama-style kwargs; `AutoModelForCausalLM.from_config` handles the config directly — no special class needed in `helpers.get_config` (and `MinistralForCausalLM` is a *different*, older architecture — don't confuse them).
- **Official checkpoints are FP8-quantized** (`config.quantization_config.quant_method == "fp8"`). Projection weights are `float8_e4m3fn` plus a `<name>.weight_scale_inv` dequant multiplier (scalar for per-tensor, 2D grid for block-wise) and an `activation_scale` (runtime-kernel-only). Casting fp8 → bf16 without multiplying by `weight_scale_inv` produces weights ~1000× too large → multilingual token-soup output with no error (`load_state_dict(strict=False)` silently ignores the scale keys). `get_shard_data` now calls `dequantize_fp8_weights` to apply the scales and drop the scale keys; norms/embeddings/head stay bf16 in the checkpoint (`modules_to_not_convert`).
- **Multimodal weight naming nests the head**: keys are `language_model.model.layers.*`, `language_model.model.embed_tokens.weight`, and `language_model.lm_head.weight`. The cache builder derives layer/embed/norm names, but `lm_head.weight` used to keep its default → `load_head` silently fell back to the embedding weights (model is untied → garbage logits). `LlmLayerCollector.__init__` now scans `layer_files` for any key ending in `lm_head.weight` when the default is missing.
- **Tokenizer needs `fix_mistral_regex=True`** (`AutoTokenizer.from_pretrained(..., fix_mistral_regex="mistralai" in model_id)` — see `end_model.py`), and in transformers 5 `apply_chat_template(..., return_tensors='pt')` returns a `BatchEncoding` unless you pass `return_dict=False`.
- Two garbled-output causes can stack: wrong lm_head *and* unscaled fp8 weights. Verify the head key resolution (`collector.lm_head_name in collector.layer_files`) and compare one dequantized weight against a manual `weight.to(bf16) * weight_scale_inv` before running end-to-end.

---

## Useful discovery commands

Fast code discovery commands that help during model bring-up:

```bash
# Find model_type dispatch points
grep -R "model_type" -n src/language_pipes/llm_layer_collector

# Find compute_head / sampling path
grep -R "def compute_head" -n src tests

# Find where a specific model class is referenced
grep -R "Gemma3" -n src/language_pipes/llm_layer_collector

# Run only the target llm_layer_collector tiny-parity test
python -m unittest tests.llm_layer_collector.test_tiny_models -k gemma3 2>&1
```

### 3. Determine expected test values

Tier 1 derives everything from your `TinyModelSpec` — you don't hand-type shapes.
For a Tier 2 `RealModelSpec` (and for sanity-checking a spec), read these from the
model's `config.json`:

| Test Value | Config Field |
|---|---|
| `hidden_size` | `config.hidden_size` |
| `vocab_size` | `config.vocab_size` |
| `head_dim` | `config.hidden_size // config.num_attention_heads` (usually) |
| `num_keys` | Total number of weight keys across all safetensor shards |

---

## LlmLayerCollector Default Parameter Reference

These defaults work for most Llama-family and Qwen-family models. If a new architecture uses different weight naming conventions, override them when constructing `LlmLayerCollector`:

| Parameter | Default | When to Change |
|---|---|---|
| `layer_prefix` | `"model.layers."` | If decoder layers use a different prefix in the state dict |
| `input_embedding_layer_name` | `"model.embed_tokens.weight"` | If embedding weight has a different key |
| `norm_layer_name` | `"model.norm.weight"` | If the final norm has a different key |
| `lm_head_name` | `"lm_head.weight"` | If the output projection has a different key |
| `shard_pattern` | `r'model-(\d+)-of-(\d+).safetensors'` | If shard files use a different naming scheme |

If the new model uses non-standard names, you would need to either:
- Pass custom values to `LlmLayerCollector` in `LlmModel.__init__()` and `EndModel.__init__()` (in `src/language_pipes/modeling/`), OR
- Ensure the model is converted/downloaded with standard naming before use.

---

## Checklist for Adding a New Model

- [ ] Identify the `model_type` string from the model's HuggingFace `config.json`
- [ ] Identify the transformers module path: `transformers.models.<name>.modeling_<name>`
- [ ] Verify the model's weight naming convention matches the defaults (or plan overrides)
- [ ] Add `DecoderLayer` to `auto/auto_layer.py` mapper
- [ ] Add `RMSNorm` (or equivalent) to `auto/auto_rms.py` mapper
- [ ] Add `RotaryEmbedding` (or equivalent) to `auto/auto_rotary.py` mapper
- [ ] Create `modeling/NewModel.py` with `compute_embedding` and `compute_layer`
- [ ] Add dispatch cases in `static_auto_model.py` for both `compute_embedding` and `compute_layer`
- [ ] Add a `TinyModelSpec` to `tests/llm_layer_collector/specs.py` and run `test_tiny_models` (optionally add a `RealModelSpec` for Tier 2)
- [ ] Run the test and verify end-to-end inference produces coherent output

## Example: How Qwen3 Was Added (Reference)

Qwen3 reuses the exact same `compute_layer` kwargs as Llama, and both use standard rotary embeddings. The `qwen3_moe` variant also reuses the same `Qwen3Model` class — it dispatches to the same code because the MoE architecture differences are handled internally by the transformers `Qwen3MoeDecoderLayer`. This is a good example of how multiple `model_type` values can share one modeling file when the computation interface is identical.
