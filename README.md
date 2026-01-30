# Language Pipes (Beta)

**A privacy focused distributed algorithm for llm inference**

[![GitHub license][License-Image]](License-Url)
[![Release][Release-Image]][Release-Url] 

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language Pipes is an open-source distributed network application designed to increase access to local language models by allowing for privacy protected computation between peer to peer nodes. 


**Disclaimer:** This software is currently in Beta. Please be patient and if you encounter an error, please [fill out a github issue](https://github.com/erinclemmer/language-pipes/issues/new)!   

---

#### Features
- Quick Setup
- OpenAI compatible API
- Privacy-focused architecture
- Decentralized peer to peer network
- Download and use models by HuggingFace ID

---

### What Does Language Pipes do?
Large Language models work by passing information through many layers. At each layer, several matrix multiplicatitons between the layer weights and the system state are performed and the data is moved to the next layer. Language pipes works by hosting different layers on different machines to split up the RAM cost across the system. This project contrasts with existing programs like vLLM by focusing on decentralization and privacy.  
  
Here are some helpful links to get started:
- To learn more about the [privacy protecting mechanism click here](./documentation/privacy.md).  
- To learn more about [how language pipes works click here](./documentation/architecture.md).  
- To learn about how Language Pipes [processes jobs click here](./documentation/job-processor.md).


### Installation
Ensure that you have Python 3.10.18 (or any 3.10 version) installed. For an easy to use Python version manager use [pyenv](https://github.com/pyenv/pyenv). This specific version is necessary for the [transformers](https://github.com/huggingface/transformers) library to work properly.  
  
If you need gpu support, first make sure you have the correct pytorch version installed for your GPU's Cuda compatibility using this link:  
https://pytorch.org/get-started/locally/

To download the models from Huggingface, ensure that you have [git](https://git-scm.com/) and [git lfs](https://git-lfs.com/) installed.  

To start using the application, install the latest version of the package from PyPi.

**Using Pip:**
```bash
pip install language-pipes
```

### Quick Start

The easiest way to get started is with the interactive setup wizard:

```bash
language-pipes
```

This launches a menu where you can create, view, and load configurations. Select **Create Config** to walk through the setup wizard, which guides you through your first configuration. After creating a config, select **Load Config** to start the server.

[We also support loading toml files directly!](./documentation/configuration.md)  
If you need help loading them [read the CLI documentation here](./documentation/cli.md).

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
| Model ID | `Qwen/Qwen3-1.7B` | HuggingFace model to load |
| Device | `cpu` | Hardware to run inference on |
| Max memory | `1` | GB of RAM to use (loads part of the model) |
| Load embedding/output layers | `Y` | Required for the first node to handle input/output |
| Enable OpenAI API | `Y` | Exposes the OpenAI-compatible endpoint |
| API port | `8000` | Port for the API server |
| First node in network | `Y` | This node starts the network |
| Encrypt network traffic | `N` | Disable encryption for simplicity |

### Node 2 (Second Computer)

Start language pipes with this command:
```bash
language-pipes
```
| Prompt | Value | Description |
|--------|-------|-------------|
| Node ID | `node-2` | Unique identifier for this node on the network |
| Model ID | `Qwen/Qwen3-1.7B` | Must match the model on node-1 |
| Device | `cpu` | Hardware to run inference on |
| Max memory | `3` | GB of RAM to use (loads remaining layers) |
| Load embedding/output layers | `N` | Node-1 already handles these |
| Enable OpenAI API | `N` | Only node-1 needs the API |
| First node in network | `N` | This node joins an existing network |
| Bootstrap node IP | `192.168.0.10` | Node-1's local IP address |
| Bootstrap port | `5000` | Node-1's network port |
| Encrypt network traffic | `N` | Must match node-1's setting |

Node-2 connects to node-1 and loads the remaining model layers. The model is now ready for inference!

### Test the API

The model is accessible via an [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/chat/create). Using the [OpenAI Python library](https://github.com/openai/openai-python):

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

To learn about how to work with the [Open AI compatable server click here](./documentation/oai.md).

## Model choice
Currently Language Pipes targets the Qwen3 and Qwen3-moe architectures.

### Dependencies
- [pytorch](pytorch.org)
- [transformers](https://huggingface.co/docs/transformers) 

### Documentation
* [CLI Reference](./documentation/cli.md)
* [Privacy Protection](./documentation/privacy.md)
* [Configuration Manual](./documentation/configuration.md)
* [Architecture Overview](./documentation/architecture.md)
* [Open AI Compatable API](./documentation/oai.md)
* [Job Processor State Machine](./documentation/job-processor.md)
* [The default peer to peer implementation](./documentation/distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./documentation/llm-layer-collector.md)
