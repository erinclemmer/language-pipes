# Configuration File Reference

This document describes the TOML configuration file format for Language Pipes.

For command-line usage, see the [CLI Reference](./cli.md).

---

## Minimal Configuration

```toml
node_id = "my-node"

[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
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
job_port = 5050
network_ip = "192.168.0.1"
bootstrap_address = "192.168.0.1"
bootstrap_port = 5000
network_key = "network.key"

# === Options ===
logging_level = "INFO"
max_pipes = 1
model_validation = true
ecdsa_verification = false
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

#### `job_port`

Port for inter-node job communication.

```toml
job_port = 5050
```

| Type | Default |
|------|---------|
| int | `5050` |

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

#### `ecdsa_verification`

Sign job packets with ECDSA. Receivers only accept packets from authorized pipes.

```toml
ecdsa_verification = true
```

| Type | Default |
|------|---------|
| bool | `false` |

---

### Directories

#### `app_dir`

Application data directory. Stores models, configs, and credentials.

```toml
app_dir = "~/.language-pipes"
```

| Type | Default |
|------|---------|
| string | `~/.language-pipes` |

**Directory structure:**
```
app_dir/
├── models/      # Downloaded model weights
├── configs/     # Configuration files
└── credentials/ # Credential files
```

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
| `job_port` | `LP_JOB_PORT` |
| `network_ip` | `LP_NETWORK_IP` |
| `bootstrap_address` | `LP_BOOTSTRAP_ADDRESS` |
| `bootstrap_port` | `LP_BOOTSTRAP_PORT` |
| `network_key` | `LP_NETWORK_KEY` |
| `max_pipes` | `LP_MAX_PIPES` |
| `model_validation` | `LP_MODEL_VALIDATION` |
| `ecdsa_verification` | `LP_ECDSA_VERIFICATION` |
| `app_dir` | `LP_APP_DIR` |

---

## See Also

- [CLI Reference](./cli.md) — Command-line usage and flags
- [Architecture](./architecture.md) — How Language Pipes works
- [OpenAI API](./oai.md) — API endpoint documentation
