# Language Pipes (Beta)

**Distribute language models across multiple systems**  

[![GitHub license][License-Image]](License-Url)
[![Release][Release-Image]][Release-Url] 

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language Pipes is a FOSS distributed network application designed to increase access to local language models.  

---  

**Disclaimer:** This software is currently in Beta. Please be patient and if you encounter an error, please [fill out a github issue](https://github.com/erinclemmer/language-pipes/issues/new)!   

Over the past few years open source language models have become much more powerful yet the most powerful models are still out of reach of the general population because of the extreme amounts of RAM that is needed to host these models. Language Pipes allows multiple computer systems to host the same model and move computation data between them so that no one computer has to hold all of the data for the model.

### Features
- Privacy-preserving architecture
- Quick Setup
- Peer to peer network
- OpenAI compatible API
- Download and use models by HuggingFace ID
- Encrypted communication between nodes

---

### What Does it do?
In a basic sense, language models work by passing information through many layers. At each layer, several matrix multiplicatitons between the layer weights and the system state are performed and the data is moved to the next layer. Language pipes works by hosting different layers on different machines to split up the RAM cost across the system.

### Installation
Ensure that you have Python 3.10.18 (or any 3.10 version) installed. For an easy to use Python version manager use [pyenv](https://github.com/pyenv/pyenv). This specific version is necessary for the [transformers](https://github.com/huggingface/transformers) library to work properly.  
  
If you need gpu support, first make sure you have the correct pytorch version installed for your GPU's Cuda compatibility using this link:  
https://pytorch.org/get-started/locally/

To download the models from Huggingface, ensure that you have [git](https://git-scm.com/) and [git lfs](https://git-lfs.com/) installed.  

To start using the application, install the latest version of the package from PyPi.

Using Pip:
```bash
pip install language-pipes
```

### Quick Start

The easiest way to get started is with the interactive setup wizard:

```bash
language-pipes
```

This launches a menu where you can create, view, and load configurations:

```
Main Menu
[0] View Config
[1] Load Config
[2] Create Config
[3] Delete Config
Select number of choice: 
```

Select **Create Config** to walk through the setup wizard, which guides you through:
- **Node ID** — A unique name for your computer on the network
- **Model selection** — Choose a HuggingFace model ID (e.g., `Qwen/Qwen3-1.7B`)
- **Device & memory** — Where to run the model and how much RAM to use
- **API server** — Enable an OpenAI-compatible endpoint
- **Network settings** — Ports and encryption options

After creating a config, select **Load Config** to start the server.

For detailed wizard documentation, see [Interactive Setup Guide](./documentation/interactive-setup.md).

---

# Two Node Example

This example shows how to distribute a model across two computers using the interactive wizard.

### Node 1 (First Computer)

```bash
language-pipes
```

1. Select **Create Config**
2. Enter a name (e.g., `node1`)
3. Follow the prompts:
   - **Node ID**: `node-1`
   - **Model ID**: `Qwen/Qwen3-1.7B` (press Enter for default)
   - **Device**: `cpu`
   - **Max memory**: `1` (loads part of the model)
   - **Load embedding/output layers**: `Y`
   - **Enable OpenAI API**: `Y`
   - **API port**: `8000`
   - **First node in network**: `Y`
   - **Encrypt network traffic**: `Y` (save the generated key!)

4. Select **Load Config** → choose `node1` to start the server

### Node 2 (Second Computer)

Install Language Pipes, then:

```bash
language-pipes
```

1. Select **Create Config**
2. Enter a name (e.g., `node2`)
3. Follow the prompts:
   - **Node ID**: `node-2`
   - **Model ID**: `Qwen/Qwen3-1.7B`
   - **Device**: `cpu`
   - **Max memory**: `3` (loads remaining layers)
   - **Load embedding/output layers**: `N` (node-1 has them)
   - **Enable OpenAI API**: `N`
   - **First node in network**: `N`
   - **Bootstrap node IP**: `192.168.0.10` (node-1's local IP)
   - **Bootstrap port**: `5000`
   - **Encrypt network traffic**: `Y`
   - **Network key**: paste the key from node-1

4. Select **Load Config** → choose `node2` to start the server

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

### Models Supported
* Llama 2 & Llama 3.X  
* Qwen3
* More to come!

### Dependencies
- [pytorch](pytorch.org)
- [transformers](https://huggingface.co/docs/transformers) 

### Documentation
* [Interactive Setup Guide](./documentation/interactive-setup.md)
* [Privacy Architecture](./documentation/privacy.md)
* [CLI Reference](./documentation/cli.md)
* [Configuration](./documentation/configuration.md)
* [Architecture](./documentation/architecture.md)
* [OpenAI API](./documentation/oai.md)
