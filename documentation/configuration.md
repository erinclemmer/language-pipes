# Configuration File Reference

This document describes the TOML configuration file format and command-line interface for Language Pipes.

---

## Commands

```
language-pipes                                  # Launch interactive TUI
language-pipes serve -c config.toml             # Run headless
language-pipes config show -c config.toml       # Print effective configuration
language-pipes config validate -c config.toml   # Validate and exit
language-pipes keygen                           # Generate AES encryption key
```

### `language-pipes`

Launches the interactive TUI. In this mode the configuration file is authoritative — environment variables and flags do not override config values. The exceptions are machine-local settings (`LP_APP_DIR`, `LP_MODEL_DIR`, `LP_HUGGINGFACE_TOKEN`, `LP_LOGGING_LEVEL`) which are respected because they describe the host environment, not the node's behavior.

### `language-pipes serve`

Runs the server without the TUI. Configuration is resolved with the following precedence (highest to lowest):

1. Environment variables (`LP_*`)
2. `--set` flags
3. Configuration file (`-c`)
4. Defaults

```
language-pipes serve -c config.toml \
  --set node_id=node-4 \
  --set peer_port=5001 \
  --log-file node4.log \
  --log-format text \
  --status-interval 30
```

| Flag | Description |
|------|-------------|
| `-c`, `--config` | Path to TOML configuration file (required) |
| `--set KEY=VALUE` | Override a config property using its TOML key name. Repeatable. |
| `--log-file PATH` | Write log output to a file in addition to stdout |
| `--log-format` | `text` (default) or `json`. Controls log output format. |
| `--log-level` | Override logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--status-interval` | Seconds between status summary lines (default: `30`) |
| `--autostart` | Skip startup confirmation and begin serving immediately |

### `language-pipes config show`

Resolves the full precedence chain (environment, `--set` flags, config file, defaults) and prints the effective configuration as valid TOML with source annotations.

```
$ LP_OAI_PORT=9000 language-pipes config show -c node4.toml

# Effective configuration
# Sources: node4.toml + environment

node_id = "node-4"                    # node4.toml
oai_port = 9000                       # LP_OAI_PORT (env)
peer_port = 5000                      # node4.toml
logging_level = "INFO"                # default
```

The output is valid TOML (comments aside) and can be piped to a file to materialize a resolved configuration:

```
language-pipes config show -c node4.toml > resolved.toml
```

### `language-pipes config validate`

Resolves configuration the same way as `config show`, validates all values, and exits with code `0` on success or `1` on failure. Useful for CI and pre-flight checks.

```
language-pipes config validate -c node4.toml
```

### `language-pipes keygen`

Generates an AES encryption key file for encrypted peer-to-peer communication.

---

## Minimal Configuration

This configuration will load "Qwen/Qwen3-1.7B" into memory with all layers and the end model, and start an OpenAI-compatible server on port 8000:

```toml
node_id = "my-node"
network_ip = "[Your local IP address]"
oai_port = 8000

[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4

end_models = ["Qwen/Qwen3-1.7B"]
```

---

## Complete Example

```toml
# === Required ===
node_id = "node-1"

[[layer_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cpu"
max_memory = 5

[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
max_memory = 8

# === End Models ===
end_models = ["meta-llama/Llama-3.2-1B-Instruct"]

# === API Server ===
oai_port = 8000
api_keys = ["test_key"]

# === Network ===
peer_port = 5000
network_ip = "192.168.0.1"
bootstrap_address = "192.168.0.2"
bootstrap_port = 5000
network_key = "network.key"

# === Options ===
logging_level = "INFO"
max_pipes = 1
model_validation = true
print_times = false
```

---

## Properties

### Required

#### `node_id`

Unique identifier for this node on the network.

```toml
node_id = "my-node-1"
```

| Type | Required |
|------|:--------:|
| string | ✓ |

#### `layer_models`

Array of models to host. Each model is defined as a TOML table.

```toml
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
```

| Property | Type | Required | Description |
|----------|------|:--------:|-------------|
| `id` | string | ✓ | HuggingFace model ID or path in `/models` directory |
| `device` | string | ✓ | PyTorch device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `max_memory` | number | ✓ | Maximum memory allocation in GB |

**Multiple models:**
```toml
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4

[[layer_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
max_memory = 8
```

#### `end_models`

Array of model IDs for which to load the End Model (embedding layer + output head). The End Model is the component that converts between text and numerical representations.

```toml
end_models = ["Qwen/Qwen3-1.7B"]
```

| Type | Default |
|------|---------|
| array of strings | `[]` (empty) |

**About End Models:**

The "ends" of a model are the embedding layer and output head — the components that convert between text and numerical representations. The node with a model in its `end_models` list is the **only node that can see your actual prompts and responses** for that model. Other nodes only process hidden state tensors and cannot read the conversation content.

```toml
# Privacy-preserving setup: you control the End Model
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 2

end_models = ["Qwen/Qwen3-1.7B"]  # Your prompts stay on this machine
```

**Multiple end models:**
```toml
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4

[[layer_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
max_memory = 8

end_models = ["Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct"]
```

---

### API Server

#### `oai_port`

Port for the [OpenAI-compatible API](./oai.md). Omit to disable the API server.

```toml
oai_port = 8000
```

| Type | Default |
|------|---------|
| int | None (disabled) |

#### `api_keys`

List of keys that are acceptable to use for the OpenAI-compatible server. [See official documentation for more information](https://developers.openai.com/api/reference/overview/).

```toml
api_keys = ["test_key"]
```

| Type | Default |
|------|---------|
| array of strings | None (disabled) |

---

### Network

These options configure the peer-to-peer network. See [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) for details.

#### `peer_port`

Port for peer-to-peer communication.

```toml
peer_port = 5000
```

| Type | Default |
|------|---------|
| int | `5000` |

#### `network_ip`

IP address that this node will advertise to other peers on the network. This is only necessary for bootstrap configurations where other nodes need to connect to this node. If not specified, the node will attempt to auto-detect its network IP.

```toml
network_ip = "192.168.1.100"
```

| Type | Default |
|------|---------|
| string | None (auto-detect) |

#### `bootstrap_address`

IP address of an existing node to join the network.

```toml
bootstrap_address = "192.168.1.100"
```

| Type | Default |
|------|---------|
| string | None |

#### `bootstrap_port`

Port of the bootstrap node.

```toml
bootstrap_port = 5000
```

| Type | Default |
|------|---------|
| int | `5000` |

#### `network_key`

Path to AES encryption key file. Generate with `language-pipes keygen`. If the value is left null then communications between nodes will not be encrypted.

```toml
network_key = "network.key"
```

| Type | Default |
|------|---------|
| string | null |

#### `whitelist_ips`

Optional list of IP addresses this node will allow for peer communication. If configured, the node will only accept inbound DSN requests from these IPs and only send outbound requests to these IPs.

```toml
whitelist_ips = ["192.168.1.100", "192.168.1.101"]
```

| Type | Default |
|------|---------|
| array of strings | `[]` (allow all IPs) |

#### `whitelist_node_ids`

Optional list of peer node IDs this node will allow for peer communication. If configured, the node will only accept inbound DSN messages from these node IDs and only send outbound DSN messages to these node IDs.

```toml
whitelist_node_ids = ["bootstrap-node", "trusted-worker-1"]
```

| Type | Default |
|------|---------|
| array of strings | `[]` (allow all node IDs) |

---

### Security

#### `model_validation`

Verify model weight hashes match other nodes on the network.

```toml
model_validation = true
```

| Type | Default |
|------|---------|
| bool | `false` |

---

### Directories

#### `app_dir`

Application configuration directory. Stores configs and credentials.

```toml
app_dir = "~/.config/language_pipes"
```

| Type | Default |
|------|---------|
| string | `~/.config/language_pipes` |

**Directory structure:**
```
app_dir/
├── configs/     # Configuration files
└── credentials/ # Credential files
```

#### `model_dir`

Application model cache directory. Stores downloaded model weights.

```toml
model_dir = "~/.cache/language_pipes/models"
```

| Type | Default |
|------|---------|
| string | `~/.cache/language_pipes/models` |

---

### Other

#### `logging_level`

Log verbosity. See [Python logging levels](https://docs.python.org/3/library/logging.html#logging-levels).

```toml
logging_level = "INFO"
```

| Type | Default | Values |
|------|---------|--------|
| string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

#### `num_local_layers`

Number of initial model layers to execute locally before forwarding work to other nodes. Higher values improve prompt obfuscation by keeping more of the early pipeline on your machine. It is expected that all nodes on the network have the same value so that model layers are loaded correctly.

```toml
num_local_layers = 1
```

| Type | Default |
|------|---------|
| int | `1` |

#### `max_pipes`

Maximum number of model pipes to participate in.

```toml
max_pipes = 2
```

| Type | Default |
|------|---------|
| int | `1` |

#### `print_times`

Print timing information for layer computations and network transfers when a job completes. Useful for debugging and performance analysis.

```toml
print_times = true
```

| Type | Default |
|------|---------|
| bool | `false` |

---

## Environment Variables

Most properties can be set via environment variables with the `LP_` prefix. In `serve` mode, environment variables take highest precedence. In TUI mode, only machine-local variables (`LP_APP_DIR`, `LP_MODEL_DIR`, `LP_HUGGINGFACE_TOKEN`, `LP_LOGGING_LEVEL`) are respected; all others are ignored.

If an environment variable is set but ignored in TUI mode, a notice is printed at startup.

| Property | Environment Variable | Respected in TUI |
|----------|---------------------|:----------------:|
| `node_id` | `LP_NODE_ID` | |
| `layer_models` | `LP_LAYER_MODELS` | |
| `logging_level` | `LP_LOGGING_LEVEL` | ✓ |
| `oai_port` | `LP_OAI_PORT` | |
| `api_keys` | `LP_API_KEYS` | |
| `peer_port` | `LP_PEER_PORT` | |
| `network_ip` | `LP_NETWORK_IP` | |
| `bootstrap_address` | `LP_BOOTSTRAP_ADDRESS` | |
| `bootstrap_port` | `LP_BOOTSTRAP_PORT` | |
| `network_key` | `LP_NETWORK_KEY` | |
| `whitelist_ips` | `LP_WHITELIST_IPS` | |
| `whitelist_node_ids` | `LP_WHITELIST_NODE_IDS` | |
| `num_local_layers` | `LP_NUM_LOCAL_LAYERS` | |
| `max_pipes` | `LP_MAX_PIPES` | |
| `model_validation` | `LP_MODEL_VALIDATION` | |
| `app_dir` | `LP_APP_DIR` | ✓ |
| `model_dir` | `LP_MODEL_DIR` | ✓ |
| `print_times` | `LP_PRINT_TIMES` | |

### Authentication (Environment Variable Only)

#### `LP_HUGGINGFACE_TOKEN`

HuggingFace API token for downloading gated or private models (like Llama). This is only available as an environment variable for security. You can also supply it at download time via a prompt. Respected in both TUI and serve modes.

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export LP_HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
language-pipes serve -c config.toml
```

**Note:** For gated models (like Llama), you must also accept the model's license agreement on the HuggingFace website before downloading.

---

## Logging Output

In `serve` mode, Language Pipes emits a line-oriented log stream to stdout (and optionally to `--log-file`). This stream contains two types of lines:

**Event lines** are emitted when something happens:
```
[14:23:04] EVENT  peer-connected node=node-3 address=192.168.0.3
[14:23:07] EVENT  model-loaded id=Qwen/Qwen3-1.7B layers=4-11 device=cpu
[14:23:09] EVENT  inference-complete job=af29 model=Qwen/Qwen3-1.7B time=1.2s
```

**Status lines** are emitted at the interval set by `--status-interval`:
```
[14:23:30] STATUS peers=3 layers=8/24 mem=2.1G reqs=147 uptime=2h13m
```

With `--log-format json`, each line is a self-contained JSON object:
```json
{"time":"14:23:04","type":"event","event":"peer-connected","node":"node-3","address":"192.168.0.3"}
{"time":"14:23:30","type":"status","peers":3,"layers":"8/24","mem":"2.1G","reqs":147,"uptime":"2h13m"}
```

All output is 80 columns or fewer and contains no ANSI escape codes, so it works cleanly with `tee`, `grep`, `tail -f`, and other standard tools:

```bash
language-pipes serve -c node4.toml | tee node4.log
```

---

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [OpenAI-Compatible API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [The default peer to peer implementation](./distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./llm-layer-collector.md)