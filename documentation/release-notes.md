# Change Log

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
Added model support for GLM4.1v, Gemma3, and Phi4 model families. Below is a list of newly tested models, the same model but . You can view current model support at [the model support page](documentation/model_support.md). Models of the same family but in different sizes should also work.

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
**--whitelist-ips** if set, only network IPs that are in this list are allowed to communicate with this node
**--whitelist-node-ids** if set, only node IDs that are in the list are allowed to communicate with this node

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

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [OpenAI-Compatible API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [Distributed State Network](./distributed-state-network/README.md)
* [LLM Layer Collector](./llm-layer-collector.md)
* [Release Notes](./release-notes.md)