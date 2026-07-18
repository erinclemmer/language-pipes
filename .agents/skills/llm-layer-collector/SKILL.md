---
name: llm-layer-collector
description: Guide for adding HuggingFace transformer model architecture support to the llm_layer_collector layer-loading and compute dispatch system.
---

# Skill: llm_layer_collector — Adding New Model Support

## Overview

`llm_layer_collector` (in `packages/llm-layer-collector/src/llm_layer_collector/`) lets Language Pipes run HuggingFace transformer architectures one decoder layer at a time. It loads weights from safetensors shards and dispatches computation by the `"model_type"` string in the model's HF `config.json` (find it via `AutoConfig.from_pretrained(id).model_type`).

Adding a model = **3 mapper entries + 2 dispatch cases + 1 new modeling file + 1 test spec entry**:

1. Register classes in `auto/auto_layer.py`, `auto/auto_rms.py`, `auto/auto_rotary.py`
2. Create `modeling/NewModel.py` with `compute_embedding` / `compute_layer`
3. Add cases to both `match` statements in `auto/static_auto_model.py`
4. Add a `TinyModelSpec` in `tests/specs.py` and run the tiny-parity suite

```
llm_layer_collector/
├── layer_collector.py    # LlmLayerCollector: shard cache, load_input_embedding/norm/head/layer_set/per_layer_embedder
├── load_layer.py         # get_shard_data (incl. fp8 dequant), load_layers → AutoDecoderLayer
├── cache.py              # shard-metadata cache (layer→file map, cache.json)
├── helpers.py            # get_config (unwraps multimodal text_config), tensor loading
├── state_obj.py          # LLmComputationState
├── auto/                 # model_type → class mappers + StaticAutoModel dispatcher
└── modeling/             # one bespoke file per architecture family
```

### Currently supported

| model_type | Modeling file | Quirks |
|---|---|---|
| `llama` | `LlamaModel.py` | baseline; simplest template |
| `phi3` | `Phi3Model.py` | sliding mask when `config.sliding_window` set |
| `qwen3` | `Qwen3Model.py` | mask keyed by `config.layer_types[layer_idx]` |
| `qwen3_moe` | `Qwen3MoeModel.py` | MoE internal to HF layer; sliding via `config.sliding_window` |
| `gemma3_text` | `Gemma3Model.py` | per-type masks+RoPE, scaled embedding, bidirectional overlay |
| `gemma4_text` | `Gemma4Model.py` | per-type masks+RoPE, PLE ride-along, `shared_kv_states` |
| `gemma4_unified_text` | `Gemma4Model.py` (shared) | like gemma4 but **no** PLE (see §9) |
| `ministral3` | `Ministral3Model.py` | fp8 checkpoints, multimodal key nesting (see §8) |
| `gpt_oss` | `GptOssModel.py` | mxfp4 experts, attention sinks — **eager only** (see §10) |

---

## How computation flows

**`StaticAutoModel.compute_embedding(prompt_tokens, chunk_size, input_embedder, input_ids, config, cache, per_layer_embedder=None)`** does the model-agnostic work: slices the next prefill chunk from `input_ids` based on `cache.get_seq_length()` (one token at a time once the prompt is consumed), embeds it, builds `cache_position`/`position_ids`, and constructs

```python
mask_kwargs = {"config": config, "inputs_embeds": hidden_state.detach(),
               "attention_mask": None, "past_key_values": cache, "position_ids": position_ids}
```

It creates an `LLmComputationState` with empty `causal_mask`/`position_embeddings` dicts, populates `state.per_layer_inputs` if a `per_layer_embedder` was passed (Gemma4 PLE — only the head node passes one), then dispatches to `<Model>.compute_embedding(state, config, mask_kwargs)`, which fills the two dicts keyed by layer type (`"full_attention"` / `"sliding_attention"`).

**`StaticAutoModel.compute_layer(layer, config, state, cache)`** dispatches to `<Model>.compute_layer`, which builds the kwargs the HF `DecoderLayer.forward()` expects and calls `layer.cls(**kwargs)`. Some modeling files take `(layer, config, state, cache)` (qwen3, phi3, qwen3_moe — they need config for layer-type lookup), others `(layer, state, cache)` (config is also reachable via `layer.cls.config`). Either works; match what your model needs.

**`StaticAutoModel.compute_head(...)`** is model-agnostic (final projection + sampling) — never needs changes.

```python
@dataclass
class LLmComputationState:                             # state_obj.py
    state: Tensor                                      # hidden states
    position_ids: Tensor
    cache_position: Tensor
    causal_mask: Dict[str, Optional[Tensor]]           # keyed by layer type; mask may be None
    position_embeddings: Dict[str, Tuple[Tensor, Tensor]]
    per_layer_inputs: Optional[Tensor] = None          # Gemma4 PLE ride-along
    shared_kv_states: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)  # Gemma4 KV sharing
```

---

## Step-by-Step: Adding a New Model

### Step 1: Register in the three auto mappers

Each of `auto/auto_layer.py`, `auto/auto_rms.py`, `auto/auto_rotary.py` has the same shape — an import plus a `mapper` dict entry:

```python
from transformers.models.<name>.modeling_<name> import NewModelDecoderLayer  # / RMSNorm / RotaryEmbedding
mapper = { ..., "new_model_type": NewModelDecoderLayer }
```

Find the classes by grepping the installed transformers source (`env/lib/python3.14/site-packages/transformers/models/<name>/modeling_<name>.py`) for `class <Name>DecoderLayer` etc. Notes:

- `AutoRMSNorm` constructs its class as `cls(config.hidden_size, eps=config.rms_norm_eps)` — verify the norm's signature is compatible.
- `AutoRotaryEmbedding(config)(x, position_ids, layer_type=None)` — pass the third arg only for per-type RoPE models (Gemma3/Gemma4).
- Registering the layer class is also what makes `supports_sdpa(config)` work: it locates the `*PreTrainedModel` class in the layer class's module and reads `_supports_sdpa`, which the collector uses to pick the attention implementation. Nothing to add per model — but check §10 if the architecture passes extra tensors to attention.

### Step 2: Create `modeling/NewModel.py`

Two static methods. The standard (Llama-style) template:

```python
class NewModel:
    @staticmethod
    def compute_embedding(state: LLmComputationState, config: PretrainedConfig,
                          mask_kwargs: Any) -> LLmComputationState:
        state.causal_mask["full_attention"] = create_causal_mask(**mask_kwargs)
        # add if the architecture has sliding-window layers:
        #   state.causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        state.position_embeddings["full_attention"] = AutoRotaryEmbedding(config)(
            state.state.detach(), state.position_ids)
        return state

    @staticmethod
    def compute_layer(layer: AutoDecoderLayer, state: LLmComputationState,
                      cache: DynamicCache) -> torch.Tensor:
        kwargs = {
            "hidden_states": state.state,
            "attention_mask": state.causal_mask["full_attention"],
            "position_embeddings": state.position_embeddings["full_attention"],
            "position_ids": state.position_ids,
            "past_key_values": cache,
        }
        return layer.cls(**kwargs)
```

The kwargs must match the HF `DecoderLayer.forward()` signature — check it in the transformers source (playbook §2). Common variations, each with an in-tree reference:

- **Per-layer-type masks and RoPE** (Gemma3/Gemma4): key both dicts by `config.layer_types[layer_idx]` and build the rotary with the `layer_type` third arg — see `Gemma3Model.py`.
- **Uniform sliding window** (Phi3, Qwen3MoE): build the sliding mask when `config.sliding_window` is set and select it for every layer — see `Phi3Model.py`.
- **Extra per-forward tensors** (Gemma4 PLE, `shared_kv_states`): see "Ride-along state" below and `Gemma4Model.py`.

### Step 3: Add dispatch cases in `auto/static_auto_model.py`

Import the class and add a `case "new_model_type":` to **both** `match` statements (`compute_embedding` and `compute_layer`). Multiple model_types can share one modeling file when the computation is identical — e.g. `case "gemma4_text" | "gemma4_unified_text":` both dispatch to `Gemma4Model`.

### Step 4: Handle config/weight-naming differences (only if non-standard)

- **Multimodal wrapper configs**: `helpers.get_config` unwraps `config.text_config` for an explicit `model_type` allowlist (`gemma3`; `gemma4`/`gemma4_unified`/`mistral3`). A new wrapper type must be added there *first* — otherwise the raw wrapper config reaches `torch.nn.Embedding(config.vocab_size, ...)` and fails with a missing-attribute error (see §9).
- **`LlmLayerCollector` weight-name defaults** (override in its constructor, called from `LlmModel.__init__`/`EndModel.__init__` in `src/language_pipes/modeling/`):

| Parameter | Default |
|---|---|
| `layer_prefix` | `"model.layers."` |
| `input_embedding_layer_name` | `"model.embed_tokens.weight"` |
| `norm_layer_name` | `"model.norm.weight"` |
| `lm_head_name` | `"lm_head.weight"` |
| `shard_pattern` | `r'model-(\d+)-of-(\d+).safetensors'` |

  The collector already auto-scans `layer_files` for any key ending in `lm_head.weight` when the default is absent, and falls back to tied embedding weights when there is no head key at all (correct for `tie_word_embeddings` models — see §9; a silent-fallback bug for untied ones — see §8).
- **Scaled word embeddings**: `load_input_embedding` special-cases Gemma3/Gemma4/Gemma4Unified to use their `*ScaledWordEmbedding` classes; a new architecture with a non-plain embedding needs the same treatment.
- **Quantized checkpoints**: handled generically in `get_shard_data`, which calls `dequantize_fp8_weights` (applies `weight_scale_inv`, drops scale keys — see §8) then `dequantize_mxfp4_weights` (unpacks `<proj>_blocks`/`<proj>_scales` into one dense `<proj>` — see §10). Both rewrite key names, so anything downstream that indexes the result by *checkpoint* key name will silently miss those tensors (see §10).
- **Attention implementation**: the collector defaults to `sdpa` only when `supports_sdpa(config)` says the architecture allows it, else `eager`. Architectures that pass extra tensors into the attention interface must run eager — §10.

### Step 5: Test

All test edits go in **one file**: `packages/llm-layer-collector/tests/specs.py`. Everything else (tiny-checkpoint building, HF parity, memory-capping) is driven off the spec tables — if a spec surfaces a gap, fix the spec table, not the test factory.

**Add one `TinyModelSpec`** (copy the closest architecture; set tiny dims + flags: `per_type_rope`, `ple`, `fp8`, `mxfp4`, `key_style`, `mrope`, `shards`). Then run — no downloads, KBs of memory:

```bash
cd packages/llm-layer-collector
python -m unittest tests.test_units tests.test_tiny_models
python -m unittest tests.test_tiny_models -k <model_type>
```

`test_tiny_models` builds a random tiny model from the HF config class, saves real safetensors shards, loads them back through `LlmLayerCollector`, and compares single-shot, chunked-prefill, and decode against the HF reference (`cosine ≥ 0.9999`, `max_abs_diff < 1e-4`). This is the primary CI gate; it catches the wiring bugs the playbook below documents. Tiny-config gotchas already encoded in specs: Phi3 needs `pad_token_id=None` (default exceeds tiny vocab); Gemma3 per-type RoPE needs `sliding_window_pattern` + `rope_local_base_freq`; Gemma4/Mistral3 specs use `key_style` to emit multimodal key nesting (`gemma4_multimodal` = `model.language_model.*`, `mistral3_multimodal` = `language_model.model.*`); gpt_oss needs `hidden_size`/`intermediate_size` as multiples of 32 (the mxfp4 block width) and an explicit `rope_parameters` whose `original_max_position_embeddings` keeps the YaRN factor consistent with the tiny context.

The `fp8` and `mxfp4` builders take opposite approaches, and the difference is the point: `fp8` quantizes real weights and mutates the reference model to the dequantized values (so rounding is applied on both sides), while `mxfp4` samples random *packed* blocks/scales and dequantizes them **into** the reference model. Sampling on the grid means the roundtrip is exact by construction, so the test measures the loader's unpacking rather than quantization error — and no quantizer has to be written or kept correct.

**Optionally add a `RealModelSpec`** (smallest real checkpoint) for the opt-in Tier 2 smoke test. Expected values derive from `config.json`/the safetensors index — never hardcoded. Runs in a memory-capped subprocess (10 GB RSS, `memguard.py`) with the layer stack streamed windowed, so default CI never downloads:

```bash
LP_RUN_MODEL_TESTS=1       python -m unittest tests.test_real_models -k <model_type>
LP_RUN_MODEL_TESTS=nightly python -m unittest tests.test_real_models   # + short greedy decode
```

Tier 2 prompts through each model's own chat template (raw `spec.prompt` only for base models without one) and decodes `spec.coherence_tokens` (default 8) greedy tokens, asserting `expected_next` appears. Set `expected_next` from *observed* output — see §9 for why guessing or copying it produces false failures.

---

## Ride-along state (extra per-forward tensors)

Some architectures need a tensor computed once at embed time that every layer node must see (reference implementation: Gemma4 Per-Layer Embeddings). Masks and RoPE already ride along in `JobData` this way. To add a new field:

1. `state_obj.py` — add an `Optional[Tensor] = None` field to `LLmComputationState`.
2. `src/language_pipes/jobs/job_data.py` — mirror the field in `JobData`; serialize in `to_bytes`/`from_bytes` (presence-flag pattern: `write_int(0)` when `None`, else `write_int(1)` + tensor bytes); handle it in `computationStateToJobData`, `jobDataToComputationState`, `detachCompState`.
3. `static_auto_model.compute_embedding` — populate it behind an optional param defaulting to `None` so other models are unaffected (PLE does this via `per_layer_embedder`).
4. `layer_collector.py` / `end_model.py` — load the producing weights **only on the head/embedding node** and null them in `clean_up`. `load_per_layer_embedder` returns `None` for non-PLE models, so layer nodes never touch the weights. This memory locality is the critical constraint: Gemma4's `embed_tokens_per_layer` is the largest tensor in the checkpoint.
5. `modeling/NewModel.compute_layer` — read the field off `state`, slicing per-layer if needed (`state.per_layer_inputs[:, :, layer_idx, :]`).

A read-only ride-along needs nothing more. If the field is **mutated as it flows** (Gemma4 cross-node KV sharing is the one in-tree case), also add write-back: `compute_layers` returns the mutated value and `Job.set_layer` writes it into `job.data` so it serializes to the next node.

---

## Debugging Playbook (test passes but output is garbled)

Section numbers are stable — `tests/specs.py` comments cite §6 and §9.

### 1) Compare hidden states against the HF reference

If generation is garbled, run the same prompt through `AutoModelForCausalLM` and through your dispatch path, and compare the last hidden state before sampling:

```bash
PYTHONPATH=packages/llm-layer-collector/src python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector import StaticAutoModel

model_dir='/home/erin/.cache/language_pipes/models/<org>/<model>/data'
cache_file='/home/erin/.cache/language_pipes/models/<org>/<model>/cache.json'

tok=AutoTokenizer.from_pretrained(model_dir)
hf=AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
ids=tok.apply_chat_template([{"role":"user","content":"How do molecules work?"}],
    tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=False)

with torch.no_grad():
    hf_last=hf.model(input_ids=ids, use_cache=True).last_hidden_state[:, -1, :].float()

collector=LlmLayerCollector(model_dir, cache_file)
emb=collector.load_input_embedding(); norm=collector.load_norm()
layers=collector.load_layer_set(0, collector.config.num_hidden_layers - 1)
cache=DynamicCache()
state=StaticAutoModel.compute_embedding(ids.shape[1], ids.shape[1], emb, ids,
    collector.config, cache, per_layer_embedder=collector.load_per_layer_embedder())
for lyr in layers:
    state.state=StaticAutoModel.compute_layer(lyr, collector.config, state, cache)
ours_last=norm(state.state)[:, -1, :].float()
print('cosine =', torch.nn.functional.cosine_similarity(hf_last, ours_last).item())  # want ~1.0
PY
```

Very low cosine (~0.1) means divergence before sampling — usually embedding/mask/positional wiring. To isolate a single component without a checkpoint, do the same with a tiny random HF model built from the config class, copying its weights into your path (this is what `test_tiny_models` automates — see `tests/synthetic.py`).

### 2) Inspect HF model internals directly

`inspect.getsource` on the exact classes reveals required kwargs, scaled embeddings, per-type RoPE, etc.:

```python
import inspect
from transformers.models.gemma3 import modeling_gemma3 as m
print(inspect.getsource(m.Gemma3DecoderLayer.forward))
```

### 3) Verify layer state-dict loading is complete

`load_layers` uses `load_state_dict(strict=False)`, so missing/renamed keys are silent. Check one layer by hand:

```bash
PYTHONPATH=packages/llm-layer-collector/src python - <<'PY'
import torch, json
from transformers import AutoConfig
from llm_layer_collector.load_layer import get_shard_data
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

model_dir='/home/erin/.cache/language_pipes/models/<org>/<model>/data'
with open(model_dir.replace('/data','/cache.json')) as f: layer_files=json.load(f)
cfg=AutoConfig.from_pretrained(model_dir)
shard=get_shard_data(0, 0, torch.device('cpu'), model_dir, 'model.layers.', layer_files, torch.bfloat16)

torch.set_default_device('meta'); lyr=AutoDecoderLayer(cfg, 0)
torch.set_default_device('cpu'); lyr=lyr.to_empty(device=torch.device('cpu'))
state={k[len('model.layers.0.'):]: v for k,v in shard.items() if k.startswith('model.layers.0.')}
res=lyr.cls.load_state_dict(state, strict=False)
print('missing:', res.missing_keys, 'unexpected:', res.unexpected_keys)  # want both empty
PY
```

### 4) NaNs partway through the layer stack → fp16 overflow

If the forward runs cleanly for several layers then goes NaN/Inf (failing index roughly consistent, shifts with prompt length), the cause is almost always **float16**. Newer architectures (Gemma3/Gemma4, some Qwen/Phi) have activation magnitudes exceeding fp16's ~65k range; residual additions accumulate until one layer overflows. Fix: load in `torch.bfloat16` (same exponent range as fp32) — and audit *every* `to(dtype=...)`/`get_shard_data(..., dtype=...)` in the load path, since a single fp16 sub-module reintroduces it. Pre-Ampere NVIDIA GPUs emulate bf16 slowly but correctly (`torch.cuda.is_bf16_supported()`); fall back to fp32 if too slow.

Quick localizer for the layer loop:

```python
comp_state.state = StaticAutoModel.compute_layer(lyr, config, comp_state, cache).detach()
if not torch.isfinite(comp_state.state).all():
    raise RuntimeError(f"NaN/Inf at layer {i}, max_abs={comp_state.state.abs().max().item()}")
```

If the failing layer is always `sliding_attention` and dtype is already bf16/fp32, suspect a `None` sliding mask before suspecting overflow.

### 5) Gemma3 gotchas

- Input embedding must be `Gemma3TextScaledWordEmbedding` (`embed_scale=sqrt(hidden_size)`), not plain `nn.Embedding` (handled in `load_input_embedding`).
- LM head is bias-free.
- When `config.use_bidirectional_attention` is set, both masks get an `or_mask_function` overlay (see `Gemma3Model.compute_embedding`).

### 6) GLM-4.1V gotchas (not currently registered; kept for bring-up)

Some GLM-4.1V checkpoints ship `"rope_scaling": null`, but `Glm4vTextAttention.forward()` unconditionally reads `self.rope_scaling["mrope_section"]` → `TypeError: 'NoneType' object is not subscriptable` at layer time. Fix: inject a default after loading the config (`RealModelSpec.inject_rope_scaling` exists for this):

```python
if config.rope_scaling is None:
    config.rope_scaling = {"rope_type": "default", "mrope_section": [16, 24, 24]}
```

Valid for `head_dim=128` because `sum([16,24,24]) * 2 == 128`. GLM also uses mrope position_ids of shape `(3, 1, seq)` — the `TinyModelSpec.mrope` flag covers this in the tiny suite.

### 7) Gemma4 gotchas

- `model_type` is `"gemma4_text"`; like Gemma3 it needs the scaled word embedding and per-type masks/RoPE keyed by `config.layer_types[layer_idx]`.
- **PLE** (active when `config.hidden_size_per_layer_input` is truthy): `compute_layer` **must** pass `per_layer_input` (the layer's slice of the `[batch, seq, num_layers, ple_dim]` ride-along) or the forward silently goes wrong. Implemented by `Gemma4PerLayerEmbedder` (in `Gemma4Model.py`, loaded via `collector.load_per_layer_embedder()` on the head node only — never on layer nodes).
- **`shared_kv_states` must be a `dict`, never `None`**: even with `num_kv_shared_layers == 0`, producer layers (last of each `layer_type`) execute `shared_kv_states[self.layer_type] = ...`. The writes are harmless when nothing reads them.
- When `num_kv_shared_layers > 0`, this dict is the one *mutated* ride-along needing write-back (see "Ride-along state"). Shared layers legitimately lack `k_proj/v_proj/k_norm/v_norm`; `strict=False` loading already tolerates that.
- MoE (`enable_moe_block`) is internal to the HF decoder layer — no special handling.
- bf16 only (fp16 overflows, §4).

### 8) Ministral3 / fp8-quantized checkpoint gotchas

- `model_type` is `"ministral3"` (text_config inside a top-level `"mistral3"` multimodal config). Standard Llama-style kwargs. (`MinistralForCausalLM` is a *different*, older architecture — don't confuse them.)
- **fp8 checkpoints** (`config.quantization_config.quant_method == "fp8"`): projection weights are `float8_e4m3fn` plus `<name>.weight_scale_inv` (scalar or block-wise 2D grid) and a runtime-only `activation_scale`. Casting fp8→bf16 *without* applying `weight_scale_inv` gives weights ~1000× too large → multilingual token-soup with no error (`strict=False` silently drops the scale keys). `dequantize_fp8_weights` in `load_layer.py` handles this; norms/embeddings/head stay bf16 (`modules_to_not_convert`).
- **Nested head key**: weights are `language_model.model.layers.*` / `language_model.lm_head.weight`. Before the `lm_head.weight`-suffix scan in `LlmLayerCollector.__init__`, `load_head` silently fell back to embedding weights — garbage logits, because this model is untied.
- Tokenizer needs `fix_mistral_regex=True` (see `end_model.py`); in transformers 5, `apply_chat_template(..., return_tensors='pt')` returns a `BatchEncoding` unless `return_dict=False`.
- Two garbled-output causes can stack (wrong head *and* unscaled fp8) — verify head key resolution and spot-check one dequantized weight before running end-to-end.

### 9) Gemma4Unified gotchas

- **A separate architecture, not a variant**: `google/gemma-4-12B-it` and up ship `model_type: "gemma4_unified"` (text: `"gemma4_unified_text"`); E2B/E4B remain plain `gemma4`. Registering a 12B checkpoint under `gemma4_text` fails immediately.
- **First symptom is a config error**: an unlisted multimodal wrapper falls through `helpers.get_config` → `'Gemma4UnifiedConfig' object has no attribute 'vocab_size'`. Add the wrapper to the allowlist first; every other symptom is downstream.
- **No PLE**: `hidden_size_per_layer_input` is 0 and the decoder layer has no `per_layer_input` param — but it takes `**kwargs`, so passing it is not a clean `TypeError`; it leaks into the attention call. `Gemma4Model.compute_layer` adds the kwarg only for `model_type == "gemma4_text"`. Everything else is shared with Gemma4, so both dispatch to `Gemma4Model`.
- **`attention_k_eq_v`** switches `full_attention` layers to `global_head_dim` + `num_global_key_value_heads` (sliding layers keep `head_dim`/`num_key_value_heads`). A tiny spec left at config defaults never exercises this — set all three `config_kwargs` to mirror the real checkpoint.
- **Tied embeddings**: no `lm_head.weight` ships, so `load_head`'s embedding fallback is *correct* here (unlike §8). The scaled embedding class stores unscaled weights (scaling happens in `forward`), so the tied head gets the right tensor. `layer_scalar` is a real per-layer param — confirm clean loading with the §3 check.
- Same `model.language_model.*` key nesting as Gemma4 (`KEY_STYLE_GEMMA4_MM`); 12B ships a single `model.safetensors`, no numbered shards.
- **Set Tier 2 `expected_next` from observed output, never copied or guessed.** Raw completion prompts make the first token a checkpoint artifact (E2B continued `' France'`, 12B emitted `'1'` — both genuine, neither meaningful), which is why Tier 2 prompts through the chat template and decodes several tokens (`coherence_tokens`, default 8 — one token only asserts `"The"`; each extra token costs a full windowed reload of the stack). Quirks are model-specific and must be measured, not assumed: gemma-4-12B always opens a thought block but still answers "Paris" within 8 tokens; Qwen3 honors `enable_thinking=False` via `template_kwargs`. Sanity-check anything degenerate against `AutoModelForCausalLM` on the same tokenization — **if HF returns the same token you produce, the dispatch is right and the expectation is wrong.**

### 10) gpt_oss / attention sinks / mxfp4 gotchas

- **`model_type` is `"gpt_oss"`**, standard Llama-style kwargs with per-layer-type masks keyed by `config.layer_types[layer_idx]` and a single (YaRN) rotary — `GptOssModel.py` is a near-copy of `Qwen3Model.py`.
- **The sdpa kernel silently drops attention sinks.** `GptOssAttention.forward` passes its sinks to the attention interface as `s_aux`; the generic `sdpa_attention_forward` accepts that into `**kwargs` and ignores it. No error, no NaN — just wrong values everywhere (observed **cosine 0.33** against HF). The architecture advertises this via `GptOssPreTrainedModel._supports_sdpa = False`, which is why the collector now asks `supports_sdpa(config)` instead of defaulting everything to sdpa.

  **Generalize the check, don't special-case the model.** `supports_sdpa` in `auto/auto_layer.py` finds the `*PreTrainedModel` class in the registered decoder layer's own module and reads the flag, so any future architecture with a sink-like extra tensor is handled at registration time with no new table. If a new model's cosine is low and the wiring looks right, **check the resolved attention implementation before anything else** — compare `collector.config._attn_implementation` against `hf.config._attn_implementation` from a real `from_pretrained`; a mismatch there explains the whole divergence.

- **mxfp4 expert weights** (`config.quantization_config.quant_method == "mxfp4"`): each fused MoE projection ships as a uint8 `<proj>_blocks` (two 4-bit values per byte, 32 per block) plus a uint8 `<proj>_scales` of per-block exponents — the dense `<proj>` key is absent. `strict=False` would drop both and run the experts on **uninitialized memory** (garbage, no error). `dequantize_mxfp4_weights` in `load_layer.py` handles it via HF's own `convert_moe_packed_tensors`.
  - A param of shape `[E, A, B]` is stored as blocks of shape `[E, B, A // 32, 16]`; the conversion transposes back at the end. Verify with the §3 clean-load check — a shape error here surfaces as `unexpected_keys`, not a crash.
  - `get_shard_data` must **not** cast blocks/scales to the compute dtype before unpacking — the bit shifts need uint8. There is an explicit suffix skip for this.
- **Dequantization renames keys, and callers outside the collector notice.** `get_avg_layer_size` (`src/language_pipes/modeling/llm_meta_data.py`) used to size a layer by looking up *checkpoint* key names in the post-dequant shard dict, inside a bare `except: pass` — so every mxfp4 expert tensor silently missed and a gpt-oss layer measured **0.054 GB instead of 1.65 GB** (30× under-count, which drives layer-to-node placement). Size from what `get_shard_data` actually returned. When adding any architecture whose weights are transformed at load time, grep for other consumers that key off checkpoint names.
- **Harmony chat format** puts the answer well past the default `coherence_tokens=8`: the template opens an `<|channel|>analysis` reasoning turn, and greedy output reaches `"Paris"` only at token 19. The spec sets `coherence_tokens=20` — measured, per §9.
- Reference parity for this model is exact (cosine 1.0000006, max_abs_diff 0.0), so treat any drift as a real regression rather than tolerance.

**General lesson**: Tier 1 passing while Tier 2 fails does *not* imply a real-checkpoint-only bug. Run the nearest untouched architecture as a control first, and treat the assertion itself as a suspect until checked against HF ground truth — the cheap discriminators come before deep debugging.

**General lesson (silent-wrong-answer class)**: the two hardest bugs here — an ignored `s_aux` and dropped mxfp4 keys — both ran cleanly and produced wrong numbers, because `**kwargs` absorption and `load_state_dict(strict=False)` are each designed to tolerate unknown keys. Neither is caught by "does it run." A **numeric parity check against `AutoModelForCausalLM` on the same tokenization is the only reliable gate**; run it before trusting any new architecture, and prefer reading a capability off the HF class (as `supports_sdpa` does) over assuming a default applies to every model.
