# Configuration

Language Pipes can be configured through TOML files, command-line arguments, or environment variables.

## Configuration Precedence

Values are resolved in this order (highest to lowest priority):

```
Command Arguments  >  Environment Variables  >  TOML Config  >  Defaults
```

---

## Quick Reference

| Property | CLI Flag | Environment Variable | Default | Required |
|----------|----------|---------------------|---------|----------|
| `node_id` | `--node-id` | `LP_NODE_ID` | - | **Yes** |
| `hosted_models` | `--hosted-models` | `LP_HOSTED_MODELS` | - | **Yes** |
| `logging_level` | `-l`, `--logging-level` | `LP_LOGGING_LEVEL` | `INFO` | No |
| `oai_port` | `--openai-port` | `LP_OAI_PORT` | None | No |
| `peer_port` | `--peer-port` | `LP_PEER_PORT` | `5000` | No |
| `job_port` | `--job-port` | `LP_JOB_PORT` | `5050` | No |
| `bootstrap_address` | `--bootstrap-address` | `LP_BOOTSTRAP_ADDRESS` | None | No |
| `bootstrap_port` | `--bootstrap-port` | `LP_BOOTSTRAP_PORT` | `5000` | No |
| `network_key` | `--network-key` | `LP_NETWORK_KEY` | `network.key` | No |
| `max_pipes` | `--max-pipes` | `LP_MAX_PIPES` | `1` | No |
| `model_validation` | `--model-validation` | `LP_MODEL_VALIDATION` | `false` | No |
| `ecdsa_verification` | `--ecdsa-verification` | `LP_ECDSA_VERIFICATION` | `false` | No |

---

## Example Configuration

```toml
# config.toml

# === Required ===
node_id = "node-1"

[[hosted_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cpu"
max_memory = 5

# === Server ===
oai_port = 6000        # OpenAI-compatible API (omit to disable)
job_port = 5050        # Job communication port

# === Network ===
peer_port = 5000
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

## Required Properties

### `node_id`

Unique identifier for this node on the network.

| | |
|---|---|
| **Type** | `string` |
| **CLI** | `--node-id` |
| **Env** | `LP_NODE_ID` |

---

### `hosted_models`

List of models to host on this node.

| | |
|---|---|
| **Type** | `array` |
| **CLI** | `--hosted-models` |
| **Env** | `LP_HOSTED_MODELS` |

#### TOML Format

```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
load_ends = false

[[hosted_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
max_memory = 8
```

#### CLI Format

Use comma-separated `key=value` pairs:

```bash
--hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=false"
```

#### Model Properties

| Key | Required | Type | Description |
|-----|:--------:|------|-------------|
| `id` | ✓ | string | HuggingFace model ID or path in `/models` |
| `device` | ✓ | string | PyTorch device (`cpu`, `cuda:0`, `cuda:1`, etc.) |
| `max_memory` | ✓ | number | Maximum memory allocation in GB |
| `load_ends` | | bool | Load embedding/head layers (default: `false`) |

---

## Server Options

### `oai_port`

Port for the [OpenAI-compatible API](./oai.md). Server is disabled if not set.

| | |
|---|---|
| **Type** | `int` |
| **Default** | None (disabled) |
| **CLI** | `--openai-port` |
| **Env** | `LP_OAI_PORT` |

---

### `job_port`

Port for job communication between nodes.

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5050` |
| **CLI** | `--job-port` |
| **Env** | `LP_JOB_PORT` |

---

### `logging_level`

Log verbosity level.

| | |
|---|---|
| **Type** | `string` |
| **Default** | `INFO` |
| **Values** | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| **CLI** | `-l`, `--logging-level` |
| **Env** | `LP_LOGGING_LEVEL` |

See [Python logging levels](https://docs.python.org/3/library/logging.html#logging-levels) for details.

---

## Network Options

These options configure the peer-to-peer network. See [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) for more information.

### `peer_port`

Port for peer-to-peer network communication.

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5000` |
| **CLI** | `--peer-port` |
| **Env** | `LP_PEER_PORT` |

---

### `bootstrap_address`

Address of an existing node to connect to when joining a network.

| | |
|---|---|
| **Type** | `string` |
| **Default** | None |
| **CLI** | `--bootstrap-address` |
| **Env** | `LP_BOOTSTRAP_ADDRESS` |

**Example:** `192.168.1.100`

---

### `bootstrap_port`

Port of the bootstrap node.

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5000` |
| **CLI** | `--bootstrap-port` |
| **Env** | `LP_BOOTSTRAP_PORT` |

---

### `network_key`

Path to the AES encryption key file for network communication.

| | |
|---|---|
| **Type** | `string` |
| **Default** | `network.key` |
| **CLI** | `--network-key` |
| **Env** | `LP_NETWORK_KEY` |

Generate a key with: `language-pipes keygen network.key`

---

## Security Options

### `model_validation`

Validate model weight hashes against other nodes to ensure model consistency.

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **CLI** | `--model-validation` |
| **Env** | `LP_MODEL_VALIDATION` |

---

### `ecdsa_verification`

Sign job packets with ECDSA so receivers only accept packets from authorized pipes.

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **CLI** | `--ecdsa-verification` |
| **Env** | `LP_ECDSA_VERIFICATION` |

---

## Other Options

### `max_pipes`

Maximum number of model pipes to participate in.

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1` |
| **CLI** | `--max-pipes` |
| **Env** | `LP_MAX_PIPES` |

---

## See Also

- [CLI Reference](./cli.md) - Command-line usage
- [Architecture](./architecture.md) - How Language Pipes works
- [OpenAI API](./oai.md) - API endpoint documentation
