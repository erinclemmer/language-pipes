```
 _                                                   ____   _
| |                                                 |  __`\(_)                
| |     __ _  ___   ___  _   _  __ _  __ _  ___     | |__) | |_ __   ___  ___ 
| |    / _` |/ _ \ / _ `| | | |/ _` |/ _` |/ _ \    |  ___/| | '_ \ / _ \/ __|
| |___| (_| | | | | (_| | |_| | (_| | (_| |  __/    | |    | | |_) |  __/\__ \
|______\__,_|_| |_|\__, |\__,_|\__,_|\__, |\___|    |_|    |_| .__/ \___||___/
                    __/ |            __/ |                   | |              
                   |___/            |___/                    |_|      
```

**Peer-to-peer distributed inference for open-source language models**

[![Release][Release-Image]][Release-Url] 
[![GitHub license][License-Image]](License-Url)
![PyPI - Downloads](https://img.shields.io/pypi/dm/language-pipes)

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language Pipes is an open-source distributed inference system built on the [transformers library](https://github.com/huggingface/transformers) that splits large language model computation across multiple machines. By separating the model's text-handling components (embedding and output head) from its intermediate transformer layers, Language Pipes enables peer-to-peer inference.

#### Features
- OpenAI-compatible API
- Interactive TUI for configuration, monitoring, and control
- Automatic model download by HuggingFace ID
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

Launch the interactive TUI:

```bash
language-pipes
```

From the main menu, select **New Configuration** and give it a name to create a TOML config and open the dashboard (or **Load Configuration** to reopen one you've created before).

The dashboard is organized into tabs along the top: **Home**, **Network**, **Models**, **Pipes**, and **Jobs**. A fresh configuration has no node ID yet, so the only option on **Home** is **Configure Network Server**. Set a Node ID under **Network > Configure**, then return to **Home** and select **Start Network Server**. Once the network is running, the dashboard exposes the rest of setup: load models under **Models > Layer Models** / **End Models**, and configure and start the OpenAI-compatible API under **Jobs > Server**.

Configuration can also be edited directly as [TOML files](./documentation/configuration.md) and run headlessly. See the [CLI reference](./documentation/cli.md) for details on running a saved configuration from the command line.

---

# Two Node Example

This example distributes `Qwen/Qwen3-1.7B` across two computers. Node 1 hosts the End Model, so prompts and responses stay on Node 1, plus enough layers to fit in its memory. Node 2 hosts the remaining layers.

### Node 1 (First Computer)

```bash
language-pipes
```

Select **New Configuration** and name it (e.g. `node-1`).

1. **Network > Configure**: set Node ID to `node-1` and enuser Network IP is set to this machine's local IP address. Leave Network Key empty to disable encryption for this example. Peer Port defaults to `5000`.
2. Back on **Home**, select **Start Network Server**.
3. **Models > Installed**: select **Install New Model** and enter `Qwen/Qwen3-1.7B` to download it.
4. **Models > Layer Models**: select **Add Layer Model**, choose `Qwen/Qwen3-1.7B`, a device (`cpu` or `cuda:0`), and a memory budget in GB (e.g. `4`), then **Save Model**. Confirm to load it now.
5. **Models > End Models**: select **Add End Model**, choose `Qwen/Qwen3-1.7B`, and confirm to load it now.
6. **Jobs > Server**: ensure the Port is set to `8000` and select **Start Server**.

### Node 2 (Second Computer)

```bash
language-pipes
```

Select **New Configuration** and name it (e.g. `node-2`).

1. **Network > Configure**: set Node ID to `node-2`. Under Bootstrap Nodes, add an entry with node-1's IP address and peer port (`5000`) so this node joins node-1's network.
2. Back on **Home**, select **Start Network Server**.
3. **Models > Installed**: install `Qwen/Qwen3-1.7B` as on Node 1.
4. **Models > Layer Models**: add `Qwen/Qwen3-1.7B` with a device and memory budget covering the remaining layers (e.g. `8` on `cuda:0`).

Once both nodes have loaded their layers, **Pipes > Complete** shows a completed pipe for `Qwen/Qwen3-1.7B`, and the model is ready for inference via node-1's Job Port.

### Test the API

The model is accessible via the [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/chat/create).  
  
Example using the [OpenAI Python library](https://github.com/openai/openai-python):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # node-1 IP address and Job Port
    api_key="not-needed"  # only required if api_keys is set in the config
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
Language Pipes currently supports a few model families including Qwen3, Phi, Meta Llama 3.1/3.2, and Gemma 3. [View all tested models here](./documentation/model_support.md)  

### Planned Improvements
- Additional model architectures
- INT8 and INT4 quantization (currently all inference uses fp16)
- GGUF format support (currently requires safetensors)

### Dependencies
- [pytorch](pytorch.org)
- [transformers](https://huggingface.co/docs/transformers) 

### Documentation
* [CLI Reference](./documentation/cli.md)
* [Privacy Protection](./documentation/privacy.md)
* [Configuration Manual](./documentation/configuration.md)
* [Architecture Overview](./documentation/architecture.md)
* [OpenAI-Compatible API](./documentation/oai.md)
* [Job Processor State Machine](./documentation/job-processor.md)
* [Distributed State Network](./documentation/distributed-state-network/README.md)
* [LLM Layer Collector](./documentation/llm-layer-collector.md)
* [Release Notes](./documentation/release-notes.md)
