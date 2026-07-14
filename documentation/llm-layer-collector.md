---
title: LLM Layer Collector
description: A Python package for loading specific components of large, sharded HuggingFace model checkpoints at the layer level.
---

A practical Python package for working with [Huggingface](huggingface.co) models at the layer level. Designed to help developers and researchers load specific model components when working with large, sharded checkpoints.

## What It Does

- Easily load layers, embedding, head, and norm and run partial computation of language models.
- Uses Huggingface file format to find the appropriate parts of the model.
- Uses the [transformers](https://github.com/huggingface/transformers) and [pytorch](pytorch.org) libraries to load data and run computations.
- Useful for research, development, and memory-constrained environments

## Essential Components

The LlmLayerCollector class serves as your central interface to the package's functionality.

#### Required Parameters:
- `model_dir`: Path to your model directory containing shards and configuration
- `cache_file`: Location for storing shard metadata

#### Optional Parameters:
- `shard_pattern`: Custom regex for matching shard files  
- `layer_prefix`: Prefix for identifying decoder layers (default: "model.layers.") 
- `input_embedding_layer_name`: Name for the embedding layer (default: 'model.embed_tokens.weight')
- `norm_layer_name`: Name for the norm weight (default: 'model.norm.weight')
- `lm_head_name`: Name for the head weight (default: 'lm_head.weight')
- `device`: Target device for tensor operations ("cpu" or "cuda") (default: "cpu")
- `dtype`: Desired numerical precision (default: torch.bfloat16)
- `load_in_8bit`: Quantize decoder layer linear weights to 8-bit with [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (LLM.int8). Requires the `bitsandbytes` package; forces `dtype` to `torch.float16` (default: False)

## Example
This example uses all of the parts of the package to generate a token prediction. The
`StaticAutoModel` helper dispatches the embedding/layer/head computation to the correct
implementation for the loaded model architecture.

```python
from llm_layer_collector import LlmLayerCollector, StaticAutoModel
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch

# Initialize core components
collector = LlmLayerCollector(
    model_dir="/path/to/model",
    cache_file="cache.json",
    device=torch.device("cuda"),
    dtype=torch.bfloat16
)

# Set up tokenization
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# Load model components
embedding = collector.load_input_embedding()
norm = collector.load_norm()
head = collector.load_head()
layers = collector.load_layer_set(0, collector.num_layers - 1)  # end layer is inclusive

# Execute forward pass
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
for layer in layers:
    state.state = StaticAutoModel.compute_layer(layer, collector.config, state, cache)

# Generate a prediction (returns the predicted token id)
next_token = StaticAutoModel.compute_head(head, norm(state.state), device="cuda", top_k=1)
print(tokenizer.decode(next_token))
```

### Computation Pipeline
The `StaticAutoModel` static methods provide a streamlined approach to model operations:
- `compute_embedding`: Embeds the input tokens, sets up the causal mask and position
  embeddings, and returns an `LLmComputationState` carrying the running hidden state
- `compute_layer`: Runs the state through a single decoder layer and returns the new hidden state
- `compute_head`: Applies the final linear projection and samples the next token id
  (supports `top_k`, `top_p`, `min_p`, and `temperature`)
