# Language Pipes (Beta)

**Distribute language models across multiple systems**  

[![GitHub license][License-Image]](License-Url)
[![Release][Release-Image]][Release-Url] 
![Discord](https://img.shields.io/discord/1406717394260594738)


[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language pipes is a FOSS distributed network application designed to increase access to local language models.  

[Join our Discord](https://discord.gg/CPvC78E53a) for any comments or questions!  

---  

**Disclaimer:** This software is currently in Beta. Please be patient and if you encounter an error, please [fill out a github issue](https://github.com/erinclemmer/language-pipes/issues/new)!   

Over the past few years open source language models have become much more powerful yet the most powerful models are still out of reach of the general population because of the extreme amounts of RAM that is needed to host these models. Language Pipes allows multiple computer systems to host the same model and move computation data between them so that no one computer has to hold all of the data for the model.
- Quick Setup
- Peer to peer network
- OpenAI compatible API
- Download and use models by HuggingFace ID
- Encrypted communication between nodes

### What Does it do?
In a basic sense, language models work by passing information through many layers. At each layer, several matrix multiplicatitons between the layer weights and the system state are performed and the data is moved to the next layer. Language pipes works by hosting different layers on different machines to split up the RAM cost across the system.

#### How is this different from Distributed Llama?
[Distributed Llama](https://github.com/b4rtaz/distributed-llama) is built to be a static network and requires individual setup and allocation for each model hosted. Language Pipes meanwhile, has a more flexible setup process that automatically selects which parts of the model to load based on what the network needs and the local systems resources. This allows separate users to collectively host a network together while maintaining trust that one configuration will not break the network. Users can come and go from the network and many different models can be hosted at the same time.

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

# Two Node Example
The following example will show how to create a small network. Firstly, create a network key for the network on the first computer:
```bash
language-pipes create_key network.key
```

Also create a `config.toml` file to tell the program how to operate:

```toml
node_id="node-1"
oai_port=6000 # Hosts an OpenAI compatible server on port 6000

[[hosted_models]]
id="Qwen/Qwen3-1.7B"
device="cpu"
max_memory=1
```

**Note:** Go to the [configuration documentation](/documentation/configuration.md) for more information about how the config properties work.

Once the configuration has been created you can start the server:
```bash
language-pipes run --config config.toml
```

This tells language pipes to download with the ID "Qwen/Qwen3-1.7B" from [huggingface.co](huggingface.co) and host it using 1GB of ram. This will load part of the model but not all of it.

Next, install the package on a separate computer on your home network and create a `config.toml` file like this:

```toml
node_id="node-2"
bootstrap_address="192.168.0.10" # Local ip address of node-1

[[hosted_models]]
id="Qwen/Qwen3-1.7B"
device="cpu"
max_memory=3
```

Copy the `network.key` file to the same directory that the config is in using a usb drive or sftp. 

Run the same command again on the computer two:
```bash
language-pipes run --config config.toml
```

Node-2 will connect to node-1 and load the remaining parts of the model. The model is ready for inference using a [standard openai chat API interface](https://platform.openai.com/docs/api-reference/chat/create). An example request to the server is provided below:

```python
import requests
import json

# node-1 IP address here
url = "http://127.0.0.1:6000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

payload = {
    "model": "Qwen/Qwen3-1.7B",
    "max_completion_tokens": 10,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about distributed systems."}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
```

### Models Supported
* Llama 2 & Llama 3.X  
* Qwen3
* More to come!

### Dependencies
- [pytorch](pytorch.org)
- [transformers](https://huggingface.co/docs/transformers) 
- [llm-layer-collector](https://github.com/erinclemmer/llm-layer-collector)
- [distributed-state-network](https://github.com/erinclemmer/distributed_state_network)

### Citation
If you use the project for an academic endeavour please use this citation.

```latex
@software{Clemmer_Language_Pipes_2025,
  author       = {Erin Clemmer},
  title        = {Language Pipes},
  abstract     = {Distribute language models across multiple systems.},
  version      = {0.0.1},
  date         = {2025-09-01},
  url          = {https://github.com/erinclemmer/language-pipes},
  keywords     = {Large Language Models, Transformers, Distributed Networks},
  license      = {MIT},
  orcid        = {0009-0005-0597-6197}
}
```