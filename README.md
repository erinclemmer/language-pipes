# Language Pipes

**Easily distribute language models across multiple systems**  

[![GitHub license][License-Image]](License-Url)
[![Release][Release-Image]][Release-Url] 

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language pipes is a distributed network application designed to increase accessability to local language models.  

---  

Over the past few years open source language models have become much more powerful yet the most powerful models are still out of reach of the general population because of the extreme amounts of RAM that is needed to host these models. Language Pipes allows multiple computer systems to host the same model and move computatiton data between them while being easy to set up.
- Quick Setup
- Peer to peer network
- OpenAI compatable API
- Download and use models by HuggingFace ID
- Encrypted communication between nodes

### What Does it do?
In a basic sense, language models work by passing information through many layers. At each layer, several matrix multiplicatitons between the layer weights and the system state are performed and the data is moved to the next layer. Language pipes works by hosting different layers on different machines to split up the RAM cost across the system.

#### How is this different from Distributed Llama?
[Distributed Llama](https://github.com/b4rtaz/distributed-llama) is built to be a static network and requires individual setup and allocation for each model hosted. Language Pipes meanwhile, has a more flexible setup process that automatically selects which parts of the model to load based on what the network needs and the local systems resources. This allows separate users to collectively host a network together while maintaining trust that one configuration will not break the network. Users can come and go from the network and many different models can be hosted at the same time.

### Quick Start
If you need gpu support, first make sure you have the correct pytorch version with this link:  
https://pytorch.org/get-started/locally/


To start using the application, install the latest version of the package from PyPi.

Using Pip:
```bash
pip install language-pipes
```

Using uv:
```bash
uv install language-pipes
```

Then, create a network key for the network:
```bash
language-pipes create_key network.key
```

Also create a `config.toml` file to tell the program how to operate. Go to the [configuration documentation](/documentation/configuration.md) for more information about how to set it up. A very simple config.toml is provided below:

```toml
node_id="node-1"
oai_port=6000

[[hosted_models]]
id="meta-llama/Llama-3.2-1B-Instruct"
device="cpu"
max_memory=1
```

Finally, start the server:
```bash
language-pipes run --config config.toml
```

This tells language pipes to download the id "meta-llama/Llama-3.2-1B-Instruct" from [huggingface.co](huggingface.co) and host it using 1GB of ram. This will load part of the model but not all of it.

Next, install the pacakge on a separate computer on your home network and create a config.toml like this:

```toml
node_id="node-2"
bootstrap_address="192.168.0.10" # IP address of node-1

[[hosted_models]]
id="meta-llama/Llama-3.2-1B-Instruct"
device="cpu"
max_memory=3
```

Run the same command again on the new computer and Node-2 will connect to node-1 and load the remaining parts of the model. The model is ready for inference using a [standard openai chat API interface](https://platform.openai.com/docs/api-reference/chat/create). An expample request to the server is provided below:

```bash
wget http://192.168.0.10:6000/v1/chat/completions \
  --header="Content-Type: application/json" \
  --post-data '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a haiku about distributed systems."}
    ]
  }' -O -
```

### Models Supported
* Llama 2 & Llama 3.X

### 

### Contributing
* PRs and collaboration are welcome :)
