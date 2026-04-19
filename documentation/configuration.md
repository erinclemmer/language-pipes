# Configuration Reference

This document describes all configuration properties available in Language Pipes. These properties can be set in a TOML configuration file, overridden with `--set KEY=VALUE` flags, or set via environment variables. See the [CLI Reference](./cli.md) for command usage and precedence rules.

---

## Minimal Configuration

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

| Type | Required | Env | Default |
|------|:--------:|-----|---------|
| string | ✓ | `LP_NODE_ID` | — |

```toml
node_id = "my-node-1"
```

#### `layer_models`

Array of models to host. Each model is defined as a TOML table.

| Env | Default |
|-----|---------|
| `LP_LAYER_MODELS` | `[]` (empty) |

```toml
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
```

Each entry has the following fields:

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `id` | string | ✓ | HuggingFace model ID or path in `/models` directory |
| `device` | string | ✓ | PyTorch device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `max_memory` | number | ✓ | Maximum memory allocation in GB |

Multiple models:
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

Array of model IDs for which to load the End Model (embedding layer + output head). The node with a model in its `end_models` list is the **only node that can see your actual prompts and responses** for that model. Other nodes only process hidden state tensors and cannot read the conversation content.

| Type | Env | Default |
|------|-----|---------|
| array of strings | — | `[]` (empty) |

```toml
end_models = ["Qwen/Qwen3-1.7B"]
```

Privacy-preserving setup:
```toml
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 2

end_models = ["Qwen/Qwen3-1.7B"]  # Your prompts stay on this machine
```

---

### API Server

#### `oai_port`

Port for the [OpenAI-compatible API](./oai.md). Omit to disable the API server.

| Type | Env | Default |
|------|-----|---------|
| int | `LP_OAI_PORT` | None (disabled) |

```toml
oai_port = 8000
```

#### `api_keys`

List of accepted API keys for the OpenAI-compatible server. [See official documentation](https://developers.openai.com/api/reference/overview/).

| Type | Env | Default |
|------|-----|---------|
| array of strings | `LP_API_KEYS` | None (disabled) |

```toml
api_keys = ["test_key"]
```

---

### Network

These options configure the peer-to-peer network. See [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) for details.

#### `peer_port`

Port for peer-to-peer communication.

| Type | Env | Default |
|------|-----|---------|
| int | `LP_PEER_PORT` | `5000` |

```toml
peer_port = 5000
```

#### `network_ip`

IP address that this node advertises to other peers. Only necessary for bootstrap configurations where other nodes connect to this node. If not specified, the node attempts to auto-detect its network IP.

| Type | Env | Default |
|------|-----|---------|
| string | `LP_NETWORK_IP` | None (auto-detect) |

```toml
network_ip = "192.168.1.100"
```

#### `bootstrap_address`

IP address of an existing node to join the network.

| Type | Env | Default |
|------|-----|---------|
| string | `LP_BOOTSTRAP_ADDRESS` | None |

```toml
bootstrap_address = "192.168.1.100"
```

#### `bootstrap_port`

Port of the bootstrap node.

| Type | Env | Default |
|------|-----|---------|
| int | `LP_BOOTSTRAP_PORT` | `5000` |

```toml
bootstrap_port = 5000
```

#### `network_key`

Path to AES encryption key file. Generate with `language-pipes keygen`. If null, peer communication is unencrypted.

| Type | Env | Default |
|------|-----|---------|
| string | `LP_NETWORK_KEY` | null |

```toml
network_key = "network.key"
```

#### `whitelist_ips`

IP addresses this node will allow for peer communication. If configured, the node only accepts inbound DSN requests from these IPs and only sends outbound requests to these IPs.

| Type | Env | Default |
|------|-----|---------|
| array of strings | `LP_WHITELIST_IPS` | `[]` (allow all) |

```toml
whitelist_ips = ["192.168.1.100", "192.168.1.101"]
```

#### `whitelist_node_ids`

Peer node IDs this node will allow for communication. If configured, the node only accepts inbound DSN messages from these node IDs and only sends outbound messages to these node IDs.

| Type | Env | Default |
|------|-----|---------|
| array of strings | `LP_WHITELIST_NODE_IDS` | `[]` (allow all) |

```toml
whitelist_node_ids = ["bootstrap-node", "trusted-worker-1"]
```

---

### Security

#### `model_validation`

Verify model weight hashes match other nodes on the network.

| Type | Env | Default |
|------|-----|---------|
| bool | `LP_MODEL_VALIDATION` | `false` |

```toml
model_validation = true
```

---

### Directories

#### `app_dir`

Application configuration directory. Stores configs and credentials.

| Type | Env | Default |
|------|-----|---------|
| string | `LP_APP_DIR` | `~/.config/language_pipes` |

```toml
app_dir = "~/.config/language_pipes"
```

Directory structure:
```
app_dir/
├── configs/     # Configuration files
└── credentials/ # Credential files
```

#### `model_dir`

Model cache directory. Stores downloaded model weights.

| Type | Env | Default |
|------|-----|---------|
| string | `LP_MODEL_DIR` | `~/.cache/language_pipes/models` |

```toml
model_dir = "~/.cache/language_pipes/models"
```

---

### Other

#### `logging_level`

Log verbosity. See [Python logging levels](https://docs.python.org/3/library/logging.html#logging-levels).

| Type | Env | Default | Values |
|------|-----|---------|--------|
| string | `LP_LOGGING_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

```toml
logging_level = "INFO"
```

#### `num_local_layers`

Number of initial model layers to execute locally before forwarding work to other nodes. Higher values improve prompt obfuscation by keeping more of the early pipeline on your machine. All nodes on the network should use the same value so that model layers are loaded correctly.

| Type | Env | Default |
|------|-----|---------|
| int | `LP_NUM_LOCAL_LAYERS` | `1` |

```toml
num_local_layers = 1
```

#### `max_pipes`

Maximum number of model pipes to participate in.

| Type | Env | Default |
|------|-----|---------|
| int | `LP_MAX_PIPES` | `1` |

```toml
max_pipes = 2
```

#### `print_times`

Print timing information for layer computations and network transfers when a job completes. Useful for debugging and performance analysis.

| Type | Env | Default |
|------|-----|---------|
| bool | `LP_PRINT_TIMES` | `false` |

```toml
print_times = true
```

---

### Authentication (Environment Variable Only)

#### `LP_HUGGINGFACE_TOKEN`

HuggingFace API token for downloading gated or private models (like Llama). Only available as an environment variable for security. You can also supply it at download time via a prompt.

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export LP_HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Note:** For gated models (like Llama), you must also accept the model's license agreement on the HuggingFace website before downloading.

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