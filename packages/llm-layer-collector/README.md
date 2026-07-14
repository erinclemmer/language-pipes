# llm-layer-collector

Load and dispatch **individual transformer layers** from HuggingFace model
checkpoints, rather than instantiating a whole model at once. Given a model
directory (config + safetensors shards), it can materialize just the embedding,
a range of decoder layers, the final norm, or the LM head — each as a
standalone `torch.nn.Module` — and run computation through them.

This powers the layer-sharding used by
[language-pipes](https://github.com/erinclemmer/language-pipes) for distributed
inference, but has no dependency on it and can be used on its own.

## Install

```bash
pip install llm-layer-collector
```

## Usage

```python
from llm_layer_collector import LlmLayerCollector

collector = LlmLayerCollector(model_dir="/path/to/model", cache_file="cache.json")
embedding = collector.load_input_embedding()
layers = collector.load_layer_set(0, 4)   # decoder layers 0..4 (end inclusive)
norm = collector.load_norm()
head = collector.load_head()
```

## Supported architectures

Llama, Phi-3, Qwen3, Qwen3-MoE, Gemma3, Gemma4, Ministral3. See
`src/llm_layer_collector/modeling/` for per-architecture support.

## License

MIT
