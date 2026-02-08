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

The "ends" of a model are the embedding layer and output head—the components that convert between text and numerical representations. The node with a model in its `end_models` list is the **only node that can see your actual prompts and responses** for that model. Other nodes only process hidden state tensors and cannot read the conversation content.

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

# Load end models for both
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

Most properties can be set via environment variables with the `LP_` prefix:

| Property | Environment Variable |
|----------|---------------------|
| `node_id` | `LP_NODE_ID` |
| `layer_models` | `LP_LAYER_MODELS` |
| `logging_level` | `LP_LOGGING_LEVEL` |
| `oai_port` | `LP_OAI_PORT` |
| `peer_port` | `LP_PEER_PORT` |
| `network_ip` | `LP_NETWORK_IP` |
| `bootstrap_address` | `LP_BOOTSTRAP_ADDRESS` |
| `bootstrap_port` | `LP_BOOTSTRAP_PORT` |
| `network_key` | `LP_NETWORK_KEY` |
| `num_local_layers` | `LP_NUM_LOCAL_LAYERS` |
| `max_pipes` | `LP_MAX_PIPES` |
| `model_validation` | `LP_MODEL_VALIDATION` |
| `app_dir` | `LP_APP_DIR` |
| `model_dir` | `LP_MODEL_DIR` |
| `print_times` | `LP_PRINT_TIMES` |

### Authentication (Environment Variable Only)

#### `LP_HUGGINGFACE_TOKEN`

HuggingFace API token for downloading gated or private models (like Llama). This is only available as an environment variable for better security. You can also supply it at download time via a prompt.

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export LP_HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
language-pipes serve -c config.toml
```

**Note:** For gated models (like Llama), you must also accept the model's license agreement on the HuggingFace website before downloading.

---

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [Open AI Compatable API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [The default peer to peer implementation](./distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./llm-layer-collector.md)
