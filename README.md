# Language Pipes

**Peer-to-peer distributed inference for open-source language models**

[![GitHub license][License-Image]](License-Url)
[![Release][Release-Image]][Release-Url] 

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language Pipes is an open-source distributed inference system built on the [transformers library](https://github.com/huggingface/transformers) that splits large language model computation across multiple machines. By separating the model's text-handling components (embedding and output head) from its intermediate transformer layers, Language Pipes enables peer-to-peer inference.

#### Features
- Interactive setup wizard
- Automatic model download by HuggingFace ID
- OpenAI-compatible API (`/v1/chat/completions`)
- Privacy-oriented architecture with layered privacy mitigations
- Decentralized peer-to-peer network with optional AES encryption

---

### How It Works
Language models process input through a sequence of transformer layers. Each layer performs matrix multiplications between learned weights and a hidden state tensor, passing the result to the next layer. Language Pipes distributes these layers across machines, splitting the memory cost across the network while keeping the text-handling components on the origin node.

The architecture provides architectural separation: layer models operate on continuous-valued tensors rather than discrete text while the end models keep text data on trusted systems. The [privacy](./documentation/privacy.md) documentation provides a probabilistic threat model that quantifies the difficulty of known inversion attacks under various mitigation configurations.

Further reading:
- [Architecture Overview](./documentation/architecture.md): runtime components and inference flow
- [Job Processor State Machine](./documentation/job-processor.md): how jobs traverse the distributed pipeline


### Installation
Requires Python 3.10+. For GPU support, install the appropriate PyTorch version for your CUDA configuration:  
https://pytorch.org/get-started/locally/  
  
**Install from pip:**
```bash
pip install language-pipes
```

### Quick Start

The easiest way to get started is with the interactive setup wizard:

```bash
language-pipes
```

This launches a menu where you can create, view, and load configurations. Select **Create Config** to walk through the setup wizard, which guides you through your first configuration. After creating a config, select **Load Config** to start the server.

Configuration can also be specified via [TOML files](./documentation/configuration.md). See the [CLI reference](./documentation/cli.md) for details on loading configurations from the command line.

---

# Two Node Example

This example shows how to distribute a model across two computers using the interactive wizard.

### Node 1 (First Computer)
Start language pipes:
```bash
language-pipes
```

| Prompt | Value | Description |
|--------|-------|-------------|
| Node ID | `node-1` | Unique identifier for this node on the network |
| Add layer models | `Y` | Starts layer model editor |
| Model ID | `Qwen/Qwen3-1.7B` | HuggingFace model to load for layers |
| Device | `cpu` | Hardware to run inference on |
| Max memory | `1` | GB of RAM to use (loads part of the model) |
| Add end model | `Y` | starts end node editor |
| Model ID | `Qwen/Qwen3-1.7B` | Huggingface model to load for ends |
| Number of local layers | 1 | Ensure first layer is loaded on your machine |
| Enable OpenAI API | `Y` | Exposes the OpenAI-compatible endpoint |
| API port | `8000` | Port for the API server |
| First node in network | `Y` | This node starts the network |
| Peer port | `5000` | Port for peer-to-peer communication |
| Network IP | `[Your local IP]` | IP address other nodes can find you at |
| Encrypt network traffic | `N` | Disable encryption for simplicity |
| Advanced options | `N` | Logging, security, etc. |

### Node 2 (Second Computer)

Start language pipes with this command:
```bash
language-pipes
```
| Prompt | Value | Description |
|--------|-------|-------------|
| Node ID | `node-2` | Unique identifier for this node on the network |
| Add layer models | `Y` | Starts layer model editor |
| Model ID | `Qwen/Qwen3-1.7B` | HuggingFace model to load for layers |
| Device | `cpu` | Hardware to run inference on |
| Max memory | `2` | GB of RAM to use (loads part of the model) |
| Add end model | `N` | starts end node editor |
| Enable OpenAI API | `Y` | Exposes the OpenAI-compatible endpoint |
| API port | `8000` | Port for the API server |
| First node in network | `N` | This node starts the network |
| Bootstrap Address | `[node-1 IP]` | IP address of node-1 |
| Peer port | `5000` | Port for peer-to-peer communication |
| Encrypt network traffic | `N` | Disable encryption for simplicity |
| Advanced options | `N` | Logging, security, etc. |

Node-2 connects to node-1 and loads the remaining model layers. The model is now distributed across both machines and ready for inference.

### Test the API

The model is accessible via the [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/chat/create).  
  
Example using the [OpenAI Python library](https://github.com/openai/openai-python):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # node-1 IP address
    api_key="not-needed"  # API key not required for Language Pipes
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    max_completion_tokens=100,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about distributed systems."}
    ]
)

print(response.choices[0].message.content)
```

Install the OpenAI library with: `pip install openai`

See the [OpenAI-compatible API documentation](./documentation/oai.md) for the full endpoint reference and sampling parameter descriptions.

### Supported Models
Language Pipes currently targets the Qwen3 and Qwen3-MoE architectures.

### Planned Improvements
- Additional model architectures
- INT8 and INT4 quantization (currently all inference uses fp16)
- GGUF format support (currently requires safetensors)
- `/v1/responses` endpoint (currently only `/v1/chat/completions`)

### Dependencies
- [pytorch](pytorch.org)
- [transformers](https://huggingface.co/docs/transformers) 

### Documentation
* [CLI Reference](./documentation/cli.md)
* [Privacy Architecture](./documentation/privacy.md)
* [SipIt Case Study](./documentation/threat-model/sipit.md)
* [Configuration Manual](./documentation/configuration.md)
* [Architecture Overview](./documentation/architecture.md)
* [OpenAI-Compatible API](./documentation/oai.md)
* [Job Processor State Machine](./documentation/job-processor.md)
* [Distributed State Network](./documentation/distributed-state-network/README.md)
* [LLM Layer Collector](./documentation/llm-layer-collector.md)
