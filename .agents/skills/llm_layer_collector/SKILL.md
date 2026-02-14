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

---

## Verifying Your Changes

### 1. Check that the model_type is recognized

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("org/model-name")
print(config.model_type)  # Should match your key in all mapper dicts
```

### 2. Run the layer collector test suite

Add a test method to `tests/llm_layer_collector/test.py` following the existing pattern:

```python
def test_new_model(self):
    model_id = "org/model-name"
    model_dir = get_model_dir(model_id)
    cache_file = get_cache_file(model_id)
    ensure_model(model_id)
    check_cache(self, model_dir, cache_file, <expected_num_keys>)
    check_embedding(self, model_dir, cache_file, 
        (<batch>, <seq_len>, <hidden_size>),  # state shape
        (<batch>, <seq_len>),                  # position_ids shape
        (<batch>, <seq_len>, <head_dim>))      # position_embeddings shape
    check_norm(self, model_dir, cache_file, <hidden_size>)
    check_head(self, model_dir, cache_file, (<vocab_size>, <hidden_size>))
    check_layers(self, model_dir, cache_file, 2)
    check_stack(self, model_dir, cache_file, chunk_size=32)
```

The `check_stack` test is the most important — it runs a full end-to-end inference (chunked prefill + decode) and verifies the model produces coherent output.

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

### 4) Gemma3-specific gotchas (learned)

- **Input embedding must be scaled** using `Gemma3TextScaledWordEmbedding` (`embed_scale=sqrt(hidden_size)`), not plain `torch.nn.Embedding`.
- **LM head should be bias-free** (`bias=False`) for this decoder-only setup.
- Matching mask APIs alone is not enough if embedding behavior differs from HF architecture.

### 5) GLM-4.1V-specific gotchas (learned)

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

# Run only the target llm_layer_collector test
python -m unittest tests.llm_layer_collector.test.TestLlmLayerCollector.test_gemma3_1B 2>&1
```

### 3. Determine expected test values

Read these from the model's `config.json`:

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
- [ ] Add a test case in `tests/llm_layer_collector/test.py`
- [ ] Run the test and verify end-to-end inference produces coherent output

## Example: How Qwen3 Was Added (Reference)

Qwen3 reuses the exact same `compute_layer` kwargs as Llama, and both use standard rotary embeddings. The `qwen3_moe` variant also reuses the same `Qwen3Model` class — it dispatches to the same code because the MoE architecture differences are handled internally by the transformers `Qwen3MoeDecoderLayer`. This is a good example of how multiple `model_type` values can share one modeling file when the computation interface is identical.
