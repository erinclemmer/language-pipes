---
title: Release Notes
description: Change log for Language Pipes releases.
---

## Release 2.4.0

### Model Support
Newly added explicit support or tested these models:

**Open AI:**
- openai/gpt-oss-20b

**Google:**
- google/gemma-4-12B-it
- google/gemma-4-31B-it
- google/gemma-4-26B-A4B-it

**Mistral AI:**
- mistralai/Ministral-3-3B-Reasoning-2512
- mistralai/Ministral-3-3B-Instruct-2512
- mistralai/Ministral-3-14B-Instruct-2512

**Meta:**
- meta-llama/Llama-3.2-3B-Instruct
- meta-llama/Llama-3.3-70B-Instruct

**Microsoft:**
- microsoft/Phi-4-mini-instruct

### Per-End-Model Configuration
End model options are now configured per model in the `end_models` list instead of globally:
- `num_local_layers` can now be set per end model (the global `LP_NUM_LOCAL_LAYERS` / `--num-local-layers` setting is deprecated and now only acts as a fallback default).
- Added a `device` option to choose the PyTorch device (`cpu`, `cuda:0`, …) used for both the local layers and the embedding/output head modules of an end model.
- The "Models / End Models" editor now includes a device selector alongside the local-layers field. See [Configuration](./configuration.md#end_models).

### Unified Logging
Unified logging across all parts of the application. Log file, "Home / Activity" page, and "language-pipes run" command should all show the same thing

### language-pipes run command
- Fixed folder initialization
- Creates a new ECDSA key if the node ID doesn't already exist.
- Now downloads model if the model in layer_models or end_models is not present on the machine.
- Added `--token` argument to run command to specify a Huggingface API token, otherwise it tries to use the global configuration value, if neither are found it downloads unauthenticated.

### Keygen command
Changed behaviour of `language-pipes keygen` command to simply print the hex value of the AES key instead of saving it to a file. This lines up with the 2.0 style of supplying the key in the config file as opposed to the config file pointing to another file.

### Bugs
- Removed requirements for logging and uuid since they come with a standard Python installation.
- Fixed selection bug in "models / layers" model editor.

## Release 2.3.0

### Standalone Packages
Two pieces of Language Pipes are now maintained as their own PyPI packages, published from this repo under [`packages/`](https://github.com/erinclemmer/language-pipes/tree/main/packages):

- **[llm-layer-collector](https://pypi.org/project/llm-layer-collector/)** — loads individual transformer components (embedding, decoder layers, norm, head) from sharded HuggingFace checkpoints and dispatches per-architecture computation. See the [package documentation](./llm-layer-collector.md).
- **[distributed-state-network](https://pypi.org/project/distributed-state-network/)** — the encrypted peer-to-peer state-sharing network that Language Pipes uses as its default router. See the [package documentation](./distributed-state-network/README.md).

Both are released at **1.0.0**. Language Pipes depends on them with exact version pins, so installing or upgrading `language-pipes` always pulls the matching versions of these packages and you never have to update them yourself.

Both packages had existed on PyPI before but I had archived them. People were still downloading them, so I've unarchived both and will be keeping them updated from this monorepo going forward. If you depend on either package directly, the `1.0.0` line is the current version.

### Bugs
- Fixed issue where job server was starting with `language-pipes run` even if the job_port property is not set.
- Logs exceptions made on frame render and does not crash app when that happens
- Fixed bug where if one node was running a model in 8 bit mode and another node was running normally it would error. Mixed precision pipes should work correctly now.
- Added keep alive message to ensure connection stays alive if preprocessing time is lengthy.

## Release 2.2.0

### Request Models From Peers
Nodes can now install a model directly from another node on the network instead of downloading it from HuggingFace. When installing a model in the TUI (**Models / Installed**) while connected to a network, you can choose between **Download from Huggingface** and the new **Request model locally** option. If a peer has the model installed, it streams the weight files to your node over the encrypted peer-to-peer channel — so on a home or lab network only one machine has to pay for the internet download.

Every transferred file is verified against the model's HuggingFace manifest by size and SHA-256 hash before it is accepted. See the [Request For Model Protocol documentation](./request-for-model.md) for how the transfer works under the hood.

### Model Support
Added support for ministral models from Mistral AI, tested Ministral-3-8B-Instruct-2512

### Tweaks
Added connected pipes information to network / peers page. 

## Release  2.1.0

### Job Limits
Added two environment variables to protect nodes from being overloaded with jobs:
- **`LP_MAX_NODE_JOBS`** (default `10`): maximum number of jobs this node will queue for a single peer node. Incoming jobs beyond this limit are rejected.
- **`LP_MAX_API_JOBS`** (default `5`): maximum number of pending jobs per API key on the OpenAI-compatible server. Requests beyond this limit are rejected until earlier jobs for that key complete.

See the [Configuration Manual](./configuration.md#environment-variables) for details.

### 8-Bit Quantization
- **`LP_8_BIT_MODE`** (default `false`): load model layers in 8-bit precision via [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (LLM.int8), roughly halving layer memory usage. Requires the `bitsandbytes` package (`pip install language-pipes[quantization]`).

### Bugs
- Fixed glitching in menu going from models / installed to models / layers
- Fixed saving and loading huggingface api key

## Release 2.0.0

### TUI
Major version change of Language Pipes, we got a TUI!

```
        _                                                   ____   _
        | |                                                 |  __`\(_)                
        | |     __ _  ___   ___  _   _  __ _  __ _  ___     | |__) | |_ __   ___  ___ 
        | |    / _` |/ _ \ / _ `| | | |/ _` |/ _` |/ _ \    |  ___/| | '_ \ / _ \/ __|
        | |___| (_| | | | | (_| | |_| | (_| | (_| |  __/    | |    | | |_) |  __/\__ \
        |______\__,_|_| |_|\__, |\__,_|\__,_|\__, |\___|    |_|    |_| .__/ \___||___/
                            __/ |             __/ |                  | |              
        Version 2.0.0      |___/             |___/                   |_|      

                                   |> New Configuration <|               

                                      Load Configuration

                                            Exit


        Arrows U/D: Move                   Enter: Select                   Esc: Back
```

The tui adds the ability to start and stop models on demand, change the configuration on the fly, and more. It's broken down into a few sections:
* **Home / Dashboard**: main status screen that allows you to start and stop the network servers as well as show the status of many different parts of the language pipes program.
* **Home / Activity**: Displays any logs for the network server, job server, or status of the models
* **Network Pages**: Change how the peer to peer network is configured for your device. Has interactive setup and you can start or stop the server at any time.
* **Model Pages**: Allows you to configure layer or end models to load on demand.
* **Pipe Pages**: Shows the status of the current pipes you are connected to or are on your network.
* **Job Pages**: Allows configuration of the job server and show the status of active jobs.

See the updated [CLI configuration guide](./cli.md) for more information on advanced usage including a `run` mode that just prints to std out.

Want to use the same tech to make your own TUI? Check out [my ansinout library](https://pypi.org/project/ansinout/)!

### Responses API
Added support for /v1/responses of the OpenAI compatable server. Works with any harness or web UI that supports the responses endpoint.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="foo"
)

response = client.responses.create(
    model="Qwen/Qwen3-1.7B",
    instructions="You are a helpful assistant.",
    input="What is distributed computing?",
    max_output_tokens=200
)

print(response.output_text)
```

### Gemma 4 Support
Added support for Google's newest language model line. Tested Models:
- google/gemma-4-E2B-it
- google/gemma-4-E4B-it
- More tests in the future, but the program should support any Gemma 4 model!

## Release 1.2.0

### --api-keys flag
Added flag to denote API keys to use for OpenAI compatible server. This flag can have many values just like `--layer-models` so use like `--api-keys foo bar baz` and then for the request add a 'Authorization' header with a value like `Bearer foo`.  
[See official documentation for more information](https://developers.openai.com/api/reference/overview/)  

#### Small changes
- Added a version checker to determine the latest version. It sends a request to raw.github.com for the pyproject.toml file and reads to version property.
- Added gemma to model support document
- Self upgrading is not a good pattern so I removed the `language-pipes upgrade` command. Use `pip install language-pipes --upgrade` to upgrade the package
- Don't default to yes for OpenAI server config

## Release 1.1.0

### Model Support
Added model support for GLM4.1v, Gemma3, and Phi4 model families. Below is a list of newly tested models, the same model but . You can view current model support at [the model support page](./model_support.md). Models of the same family but in different sizes should also work.

**Phi:**
- microsoft/Phi-4-mini-reasoning
- microsoft/Phi-4-reasoning-plus

**Z.ai:**
- zai-org/GLM-4.1V-9B-Thinking

**Meta:**
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-1B-Instruct

**Gemma:**
- google/gemma-3-1b-it

### Whitelists
**--whitelist-node-ids** if set, only node IDs that are in the list are allowed to communicate with this node

> **Note:** The `whitelist-ips` feature has been dropped in favor of `whitelist-node-ids`. IP addresses are not a stable, authenticated peer identity, so IP-based whitelisting has been removed; use node-ID whitelisting instead.

**Note:** `--whitelist-node-ids` works by ECDSA signature. Public keys are stored in `~/.config/language_pipes/credentials` and returning nodes must match the same public signature to be allowed to use their node ID.

### Bug fixes
- Asks for huggingface API token via text prompt if not in environment variable.
- Models layers don't have to line up exactly to complete a pipe, this should make `num_local_layers` easier to work with.
- `v1/models` endpoint now only includes a model if all of the layers are available in a pipe AND the node hosting the open AI compatible server has an end model for that model ID
- Added more stop tokens from the config. Llama3 does not use the same place so we pass a set of tokens as stop tokens instead of just `config.eos_token_id`.

## Release 1.0.0

### Features:
**Huggingface hub**  
We are now using the `huggingface_hub` python library to download models. This allows you to download gated models by supplying it an api key either through the `LP_HUGGINGFACE_TOKEN` environment variable or through input whenever a model needs to be downloaded.

**--end-models**  
Added a `--end-models` flag for setting end models to run, also available as a configuration setting or environment variable. This changes the configuration from the `--hosted-models` flag to separate `--layer-models` and `--end-models` flags. You can use the flag as follows
```bash
language-pipes serve --config config.toml --end-models [model name]
```

**--num-local-layers**
Adds the `--num-local-layers` flag for a new layer of security for end models. Also available in a configuration setting or environment variable. This setting lets you set how many layers an end model will run. This will mitigate SipIt attacks (see the [write up here](https://github.com/erinclemmer/language-pipes/blob/main/documentation/threat-model/sipit.md)). **Important** this setting must be the same for all nodes on the network. You can use the flag as follows:
```bash
language-pipes serve --config config.toml --num-local-layers 5
```

**AES keys**
We now support rolling AES IVs to prevent [predictable IV attacks](https://cwe.mitre.org/data/definitions/329.html)  

**Layer size computation**  
Uses a better algorithm for detecting the size of the weights inside of a layer.

**Usability**  
The project is using the keyboard interrupt (ctrl+c) as a "go back" button to go back to a previous menu. This has been overhauled and more places should allow you to go back to the last menu instead of exiting the program or going back to the main menu. You can always use the ctrl+d interrupt to kill the process.

**Logging**  
Added the date and time to the logger for better logs on long running servers.

### Notes
I'm finally happy with the way the project is laid out and what it is trying to accomplish. Privacy for layer nodes is a much more complicated problem that I first realized so I have updated the privacy wording in the documentation to match a probabilistic threat model instead of assuming hidden states can never be reversed. I will be doing case studies in hidden state reversal and finding mitigation techniques that give the user more trust that their prompts will not be compromised.
