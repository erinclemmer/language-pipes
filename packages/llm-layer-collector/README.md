# llm-layer-collector

`llm_layer_collector` is a Python package for work with [HuggingFace](https://huggingface.co) models at the layer level. The package loads the embedding, the decoder layers, the norm, and the head as separate PyTorch modules. A program can thus load only the parts of a model that it needs. This is useful for research, for development, and for machines with a small quantity of memory.

Language Pipes uses this package for distributed inference. The package has no dependency on Language Pipes and can operate alone.

## What the package does

- It reads the HuggingFace file format to find the correct parts of a checkpoint.
- It loads each part as a standard [PyTorch](https://pytorch.org) module.
- It runs the computation for each part with the [transformers](https://github.com/huggingface/transformers) library.
- It selects the correct computation for the architecture of the loaded model.

## Installation

```bash
pip install llm-layer-collector
```

## Public interface

The package makes two classes available at the top level:

```python
from llm_layer_collector import LlmLayerCollector, StaticAutoModel
```

| Name | Type | Function |
|---|---|---|
| `LlmLayerCollector` | Class | Reads the checkpoint and loads the model parts. |
| `StaticAutoModel` | Class with static methods only | Runs the computation for the loaded model parts. |

Four more classes come back from the methods of these two classes. A program does not construct these classes directly, but a program does use their methods:

| Name | Module | Function |
|---|---|---|
| `LLmComputationState` | `llm_layer_collector.state_obj` | Holds the hidden state and the position data between the steps. |
| `AutoDecoderLayer` | `llm_layer_collector.auto.auto_layer` | Wraps one decoder layer of the applicable architecture. |
| `AutoRMSNorm` | `llm_layer_collector.auto.auto_rms` | Wraps the final norm of the applicable architecture. |
| `Gemma4PerLayerEmbedder` | `llm_layer_collector.modeling.Gemma4Model` | Computes the Per-Layer Embeddings (PLE) for Gemma4. |

---

## `LlmLayerCollector`

The `LlmLayerCollector` class is the central interface to the package. The constructor reads `config.json` from the model directory. Then the constructor reads the cache file, or builds a new cache file. The cache file holds a map from each tensor name to the shard file that contains the tensor.

### Constructor

```python
LlmLayerCollector(
    model_dir,
    cache_file,
    shard_pattern=r"model-(\d+)-of-(\d+).safetensors",
    layer_prefix="model.layers.",
    input_embedding_layer_name="model.embed_tokens.weight",
    norm_layer_name="model.norm.weight",
    lm_head_name="lm_head.weight",
    dtype=torch.bfloat16,
    device=torch.device("cpu"),
    load_in_8bit=False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_dir` | `Path` | — | The path to the model directory. The directory must contain `config.json` and the shard files. |
| `cache_file` | `Path` | — | The path to the cache file for the shard data. This parameter is necessary. The constructor raises an exception if the value is `None`. |
| `shard_pattern` | `str` | `model-(\d+)-of-(\d+).safetensors` | A regular expression that matches the shard files. |
| `layer_prefix` | `str` | `model.layers.` | The prefix of the names of the decoder layer tensors. |
| `input_embedding_layer_name` | `str` | `model.embed_tokens.weight` | The name of the tensor for the input embedding. |
| `norm_layer_name` | `str` | `model.norm.weight` | The name of the tensor for the final norm. |
| `lm_head_name` | `str` | `lm_head.weight` | The name of the tensor for the head. |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The numerical precision of the loaded tensors. |
| `device` | `torch.device` | `torch.device("cpu")` | The default device for the loaded modules. |
| `load_in_8bit` | `bool` | `False` | Quantizes the linear weights of the decoder layers to 8 bits with [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (LLM.int8). |

The constructor corrects three of these values automatically:

- If the constructor builds a new cache file, it reads the true `layer_prefix`, `input_embedding_layer_name`, and `norm_layer_name` from the tensor names in the shards. Thus a program does not usually set these three parameters.
- If `lm_head_name` is not in the cache, but a different name ends with `lm_head.weight`, the constructor uses that name. Multimodal checkpoints nest the head under a different prefix.
- If `load_in_8bit` is `True`, the constructor sets `dtype` to `torch.float16`. The bitsandbytes kernels compute in fp16, so the other parts must have the same type.

**CAUTION:** Delete the cache file after you change or replace the files in the model directory. A stale cache file points to shard files that are no longer correct, and the load then fails or gives incorrect weights.

**NOTE:** The `load_in_8bit` option needs the `bitsandbytes` package. The load of the first layer set raises an `ImportError` if the package is not installed.

### Attributes

The constructor sets these public attributes:

| Attribute | Type | Description |
|---|---|---|
| `config` | `PretrainedConfig` | The configuration of the model. For a multimodal checkpoint, this is the text configuration. |
| `num_layers` | `int` | The number of decoder layers in the model. |
| `layer_files` | `Dict[str, str]` | A map from each tensor name to the name of its shard file. |
| `model_dir` | `Path` | The model directory that the constructor received. |
| `cache_file` | `Path` | The cache file that the constructor received. |
| `dtype` | `torch.dtype` | The precision in use. This can be different from the `dtype` parameter (refer to the previous section). |
| `device` | `torch.device` | The default device for the loaded modules. |
| `load_in_8bit` | `bool` | Shows if 8-bit quantization is active. |
| `layer_prefix`, `input_embedding_layer_name`, `norm_layer_name`, `lm_head_name`, `shard_pattern` | `str` | The tensor names in use after the automatic correction. |

Each load method has an optional `device` parameter. If the value is `None`, the method uses the `device` attribute.

### `load_input_embedding(device=None)`

**Returns:** `torch.nn.Embedding`

Loads the weight of the input embedding and gives back an embedding module. For the Gemma3 and Gemma4 architectures, the method gives back the scaled embedding class of that architecture. For all other architectures, the method gives back a standard `torch.nn.Embedding`.

```python
embedding = collector.load_input_embedding()
```

### `load_norm(device=None)`

**Returns:** `AutoRMSNorm`

Loads the weight of the final norm and gives back an `AutoRMSNorm`. The `AutoRMSNorm` object contains the RMS norm class of the applicable architecture. Call the object directly to apply the norm to a hidden state:

```python
norm = collector.load_norm()
normed_state = norm(state.state)
```

### `load_head(device=None)`

**Returns:** `torch.nn.Linear`

Loads the weight of the head and gives back a linear module without a bias. If the checkpoint has no separate head tensor, the method uses the weight of the input embedding. Models with tied weights keep the head and the embedding in one tensor.

```python
head = collector.load_head()
```

### `load_layer_set(start_layer, end_layer, device=None)`

**Returns:** `List[AutoDecoderLayer]`

Loads a continuous set of decoder layers. The method loads the layers in groups of three, and calls the garbage collector at the end. This procedure keeps the peak memory low for large models.

| Parameter | Type | Description |
|---|---|---|
| `start_layer` | `int` | The index of the first layer. |
| `end_layer` | `int` | The index of the last layer. This layer is part of the result. |
| `device` | `Optional[torch.device]` | The device for the layers. |

**CAUTION:** The `end_layer` index is inclusive. To load all layers of a model, give `collector.num_layers - 1` as the value. A value of `collector.num_layers` raises an exception, because there is no data for that layer.

```python
# All layers of the model
layers = collector.load_layer_set(0, collector.num_layers - 1)

# Only layers 4 to 8 (five layers)
layers = collector.load_layer_set(4, 8)
```

The method also converts the quantized weights of the checkpoint:
- It applies the fp8 scales to their weights.
- It unpacks the mxfp4 expert weights of the MoE models.
- It fuses the per-expert weights into the stacked tensors that the transformers MoE modules use.

### `load_per_layer_embedder(device=None)`

**Returns:** `Optional[Gemma4PerLayerEmbedder]`

Loads the three Per-Layer Embedding (PLE) weights of Gemma4. The method gives back `None` for each model that does not use PLE. Give the result to `StaticAutoModel.compute_embedding()` as the `per_layer_embedder` parameter.

**CAUTION:** Call this method only on the node that holds the embedding and the head. The `embed_tokens_per_layer` tensor is the largest single tensor in the checkpoint, and a load on a layer node can fill the memory of that node.

```python
per_layer_embedder = collector.load_per_layer_embedder()
```

---

## `StaticAutoModel`

The `StaticAutoModel` class has three static methods. Each method sends the computation to the implementation for the architecture of the loaded model. The class holds no state, so a program does not construct it.

### `compute_embedding(...)`

```python
StaticAutoModel.compute_embedding(
    prompt_tokens,
    chunk_size,
    input_embedder,
    input_ids,
    config,
    cache,
    per_layer_embedder=None,
)
```

**Returns:** `LLmComputationState`

Embeds the next tokens and prepares the data that the decoder layers need. The method selects the tokens with the cache: it starts at the number of tokens that are already in the cache. If prompt tokens remain, the method takes a maximum of `chunk_size` tokens. If no prompt tokens remain, the method takes one token. The one-token path is the decode step.

The method then computes the causal mask and the rotary position embeddings for the architecture. All results go into the `LLmComputationState` object.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt_tokens` | `int` | — | The number of tokens in the prompt. |
| `chunk_size` | `int` | — | The maximum number of tokens for one prefill chunk. |
| `input_embedder` | `torch.nn.Embedding` | — | The embedding module from `load_input_embedding()`. |
| `input_ids` | `torch.Tensor` | — | The token ids of the full prompt. |
| `config` | `PretrainedConfig` | — | The configuration from `collector.config`. |
| `cache` | `DynamicCache` | — | The key-value cache of the job. |
| `per_layer_embedder` | `Optional[torch.nn.Module]` | `None` | The Gemma4 PLE module from `load_per_layer_embedder()`. |

**NOTE:** For a prompt with no chunks, set `chunk_size` to the value of `prompt_tokens`.

### `compute_layer(layer, config, state, cache)`

**Returns:** `torch.Tensor`

Runs the hidden state through one decoder layer and gives back the new hidden state. The method does not change the `state` object, so the caller must write the result to `state.state` before the next layer.

| Parameter | Type | Description |
|---|---|---|
| `layer` | `AutoDecoderLayer` | One layer from `load_layer_set()`. |
| `config` | `PretrainedConfig` | The configuration from `collector.config`. |
| `state` | `LLmComputationState` | The state from `compute_embedding()`. |
| `cache` | `DynamicCache` | The same cache object that `compute_embedding()` received. |

```python
for layer in layers:
    state.state = StaticAutoModel.compute_layer(layer, collector.config, state, cache)
```

**NOTE:** The method gives back an empty tensor if the architecture of the layer is not supported.

### `compute_head(head, state, device, top_k=1, top_p=1, min_p=0, temperature=1)`

**Returns:** `int` — the id of the next token.

Applies the head projection to the last position of the hidden state, and then selects the next token. Apply the final norm to the hidden state before you call this method.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `head` | `torch.nn.Linear` | — | The head module from `load_head()`. |
| `state` | `torch.Tensor` | — | The hidden state after the final norm. |
| `device` | `str` | — | The device for the projection, for example `"cuda"`. |
| `top_k` | `int` | `1` | Keeps only the `top_k` tokens with the highest logits. A value of 0 stops this filter. |
| `top_p` | `float` | `1` | Keeps the tokens with the highest probability until the sum is more than `top_p`. A value of 1 stops this filter. |
| `min_p` | `float` | `0` | Removes each token with a probability less than `min_p` multiplied by the highest probability. A value of 0 stops this filter. |
| `temperature` | `float` | `1` | Divides the logits. A low value makes the distribution sharp. A high value makes the distribution flat. |

The method uses one of two paths:

1. If `temperature` is 0, the method selects the token with the highest logit. This path is greedy decoding, and it uses no filter.
2. If `temperature` is not 0, the method divides the logits by `temperature`. Then the method applies the `min_p`, `top_p`, and `top_k` filters in that sequence. At the end, the method samples one token from the result.

```python
next_token = StaticAutoModel.compute_head(head, norm(state.state), device="cuda", top_k=1)
```

---

## `LLmComputationState`

The `LLmComputationState` dataclass holds the data that moves between the embedding, the layers, and the head. `compute_embedding()` constructs the object, and `compute_layer()` reads the object.

| Field | Type | Description |
|---|---|---|
| `state` | `Tensor` | The hidden state. The caller updates this field after each layer. |
| `position_ids` | `Tensor` | The position index of each token in the current chunk. |
| `cache_position` | `Tensor` | The position of each token in the full sequence. |
| `causal_mask` | `Dict[str, Optional[Tensor]]` | The attention masks for each mask type of the architecture. |
| `position_embeddings` | `Dict[str, Tuple[Tensor, Tensor]]` | The cosine and sine tensors of the rotary embeddings. |
| `per_layer_inputs` | `Optional[Tensor]` | The Gemma4 PLE tensor, or `None`. |
| `shared_kv_states` | `Dict[str, Tuple[Tensor, Tensor]]` | The key-value states that more than one layer shares. |

---

## Full example

This example loads all parts of a model and predicts one token.

```python
from llm_layer_collector import LlmLayerCollector, StaticAutoModel
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch

# 1. Construct the collector.
collector = LlmLayerCollector(
    model_dir="/path/to/model",
    cache_file="cache.json",
    device=torch.device("cuda"),
    dtype=torch.bfloat16
)

# 2. Tokenize the prompt.
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# 3. Load the model parts.
embedding = collector.load_input_embedding()
norm = collector.load_norm()
head = collector.load_head()
layers = collector.load_layer_set(0, collector.num_layers - 1)  # end layer is inclusive

# 4. Compute the embedding.
cache = DynamicCache()
prompt_tokens = input_ids.shape[1]
state = StaticAutoModel.compute_embedding(
    prompt_tokens=prompt_tokens,
    chunk_size=prompt_tokens,
    input_embedder=embedding,
    input_ids=input_ids,
    config=collector.config,
    cache=cache,
)

# 5. Run the state through each layer.
for layer in layers:
    state.state = StaticAutoModel.compute_layer(layer, collector.config, state, cache)

# 6. Apply the norm and the head to get the next token.
next_token = StaticAutoModel.compute_head(head, norm(state.state), device="cuda", top_k=1)
print(tokenizer.decode(next_token))
```

To do the same task step by step:

1. Construct an `LlmLayerCollector` with the model directory and a cache file path.
2. Tokenize the prompt with the tokenizer of the model.
3. Load the embedding, the norm, the head, and the layer set.
4. Construct a `DynamicCache`.
5. Call `StaticAutoModel.compute_embedding()` to get the first `LLmComputationState`.
6. Call `StaticAutoModel.compute_layer()` for each layer. Write each result to `state.state`.
7. Apply the norm to `state.state`.
8. Call `StaticAutoModel.compute_head()` with the normed state to get the next token id.
9. To get more tokens, add the new token id to `input_ids` and do steps 5 to 8 again. Use the same cache object.

---

## Supported architectures

The three `StaticAutoModel` methods and the layer classes support these model types:

| `config.model_type` | Model family |
|---|---|
| `llama` | Llama |
| `phi3` | Phi-3 and Phi-4 |
| `qwen3` | Qwen3 |
| `qwen3_moe` | Qwen3 MoE |
| `gemma3_text` | Gemma 3 |
| `gemma4_text` | Gemma 4 |
| `gemma4_unified_text` | Gemma 4 Unified |
| `ministral3` | Ministral 3 |
| `gpt_oss` | GPT-OSS |

For a multimodal checkpoint, the collector reads the text configuration. Thus a model with the type `gemma3` becomes `gemma3_text`.

To add a new architecture, refer to the modules in `src/llm_layer_collector/modeling/`.
