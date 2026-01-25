# LLM Layer Collector

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
- `norm_layer_name`: Name for the norm weight (default: 'momdel.norm.weight')
- `lm_head_name`: Name for the head weight (default: 'lm_head.weight')
- `device`: Target device for tensor operations ("cpu" or "cuda") (default: "cpu")
- `dtype`: Desired numerical precision (default: torch.float16)

## Example
This example uses all of the parts of the package to generate a token prediction

```python
from llm_layer_collector import LlmLayerCollector
from llm_layer_collector.compute import compute_embedding, compute_layer, compute_head
from transformers import AutoTokenizer
import torch

# Initialize core components
collector = LlmLayerCollector(
    model_dir="/path/to/model",
    cache_file="cache.json",
    device="cuda",
    dtype=torch.float16
)

# Set up tokenization
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# Load model components
embedding = collector.load_input_embedding()
norm = collector.load_norm()
head = collector.load_head()
layers = collector.load_layer_set(0, collector.num_layers - 1)

# Execute forward pass
state = compute_embedding(embedding, input_ids, collector.config)
for layer in layers:
    state.state = compute_layer(layer, state)

# Generate predictions
predictions = compute_head(head, norm(state.state), topk=1)
```

### Computation Pipeline
Our helper functions provide a streamlined approach to model operations:
- `compute_embedding`: Handles input embedding and causal mask setup
- `compute_layer`: Manages state transitions through decoder layers
- `compute_head`: Processes final linear projections and token prediction
