# Configuration File Reference

This document describes the TOML configuration file format for Language Pipes.

For command-line usage, see the [CLI Reference](./cli.md).

---

## Minimal Configuration

These configuration options will:
- Load "Qwen/Qwen3-1.7B" into memory with all layers and the end model
- Start an Open AI compatable server on port 8000

```toml
node_id = "my-node"
network_ip = "[Your local IP address]"
openai_port = 8000

[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
load_ends = true
```

---

## Complete Example

```toml
# === Required ===
node_id = "node-1"

[[hosted_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cpu"
max_memory = 5
load_ends = true

[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
max_memory = 8

# === API Server ===
oai_port = 8000

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

#### `hosted_models`

Array of models to host. Each model is defined as a TOML table.

```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
load_ends = false
```

| Property | Type | Required | Description |
|----------|------|:--------:|-------------|
| `id` | string | ✓ | HuggingFace model ID or path in `/models` directory |
| `device` | string | ✓ | PyTorch device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `max_memory` | number | ✓ | Maximum memory allocation in GB |
| `load_ends` | bool | | Load the End Model: embedding + output head (default: `false`) |

**About `load_ends` (End Model):**

The "ends" of a model are the embedding layer and output head—the components that convert between text and numerical representations. The node with `load_ends = true` is the **only node that can see your actual prompts and responses**. Other nodes only process hidden state tensors and cannot read the conversation content.

```toml
# Privacy-preserving setup: you control the End Model
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
load_ends = true   # Your prompts stay on this machine
max_memory = 2
```

For maximum privacy, enable `load_ends` on your own machine and let untrusted nodes contribute compute with `load_ends = false`.

**Multiple models:**
```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4

[[hosted_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
max_memory = 8
load_ends = true
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

#### `logging_level`

Log verbosity. See [Python logging levels](https://docs.python.org/3/library/logging.html#logging-levels).

```toml
logging_level = "INFO"
```

| Type | Default | Values |
|------|---------|--------|
| string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

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

All properties can be set via environment variables with the `LP_` prefix:

| Property | Environment Variable |
|----------|---------------------|
| `node_id` | `LP_NODE_ID` |
| `hosted_models` | `LP_HOSTED_MODELS` |
| `logging_level` | `LP_LOGGING_LEVEL` |
| `oai_port` | `LP_OAI_PORT` |
| `peer_port` | `LP_PEER_PORT` |
| `network_ip` | `LP_NETWORK_IP` |
| `bootstrap_address` | `LP_BOOTSTRAP_ADDRESS` |
| `bootstrap_port` | `LP_BOOTSTRAP_PORT` |
| `network_key` | `LP_NETWORK_KEY` |
| `max_pipes` | `LP_MAX_PIPES` |
| `model_validation` | `LP_MODEL_VALIDATION` |
| `app_dir` | `LP_APP_DIR` |
| `model_dir` | `LP_MODEL_DIR` |
| `print_times` | `LP_PRINT_TIMES` |

---

## See Also

- [CLI Reference](./cli.md) — Command-line usage and flags
- [Architecture](./architecture.md) — How Language Pipes works
- [JobProcessor State Machine](./job-processor.md) — Job processing FSM details
- [OpenAI API](./oai.md) — API endpoint documentation
