---
title: Configuration Reference
description: The TOML configuration file format used by Language Pipes (LpConfig) — every field, type, and default.
---

This document describes the TOML configuration file format used by Language
Pipes (`LpConfig`). See the [CLI Reference](./cli.md) for how `-c`/`--config`
resolves a configuration file and how it is used by `run`, `config`, and the
TUI.

A small set of machine-local settings are controlled by environment variables
instead of the TOML file. See [Environment Variables](#environment-variables)
below.

---

## Minimal Configuration

```toml
node_id = "my-node"
job_port = 8000
end_models = ["Qwen/Qwen3-1.7B"]

[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cpu"
memory = 4
```

> **Key ordering matters.** In TOML, key/value pairs that appear after an
> array-of-tables header (`[[layer_models]]`, `[[bootstrap_nodes]]`) belong to
> that table, not to the top-level document. Put all top-level scalar keys
> (`node_id`, `job_port`, `end_models`, etc.) **before** any `[[layer_models]]`
> or `[[bootstrap_nodes]]` blocks.

## Complete Example

```toml
# === Required ===
node_id = "node-1"

# === End Models ===
end_models = ["meta-llama/Llama-3.2-1B-Instruct"]

# === API Server ===
job_port = 8000
api_keys = ["test_key"]

# === Network ===
peer_port = 5000
network_ip = "192.168.0.1"
network_key = "9f86d081884c7d659a2feaa0c55ad015"
whitelist_ips = []
whitelist_node_ids = []

# === Layer Models ===
[[layer_models]]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cpu"
memory = 5

[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
memory = 8

# === Bootstrap Nodes ===
[[bootstrap_nodes]]
address = "192.168.0.2"
port = 5000
```

---

## Properties

### Required

#### `node_id`

Unique identifier for this node on the network.

| Type | Required | Default |
|------|:--------:|---------|
| string | ✓ | — |

```toml
node_id = "my-node-1"
```

#### `layer_models`

Array of models to host. Each model is defined as a TOML table.

| Type | Default |
|------|---------|
| array of tables | `[]` (empty) |

```toml
[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cpu"
memory = 4
```

Each entry has the following fields:

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `model_id` | string | ✓ | HuggingFace model ID or path in `/models` directory |
| `device` | string | ✓ | PyTorch device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `memory` | number | ✓ | Maximum memory allocation in GB |

Multiple models:
```toml
[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cpu"
memory = 4

[[layer_models]]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
memory = 8
```

#### `end_models`

Array of model IDs for which to load the End Model (embedding layer + output head). The node with a model in its `end_models` list is the **only node that can see your actual prompts and responses** for that model. Other nodes only process hidden state tensors and cannot read the conversation content.

| Type | Default |
|------|---------|
| array of strings | `[]` (empty) |

```toml
end_models = ["Qwen/Qwen3-1.7B"]
```

Privacy-preserving setup:
```toml
end_models = ["Qwen/Qwen3-1.7B"]  # Your prompts stay on this machine

[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cpu"
memory = 2
```

---

### API Server

#### `job_port`

Port for the [OpenAI-compatible API](./oai.md). Omit to disable the API server.

| Type | Default |
|------|---------|
| int | None (disabled) |

```toml
job_port = 8000
```

#### `api_keys`

List of accepted API keys for the OpenAI-compatible server.

| Type | Default |
|------|---------|
| array of strings | None (disabled) |

```toml
api_keys = ["test_key"]
```

---

### Network

These options configure the peer-to-peer network. See [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) for details.

#### `peer_port`

Port for peer-to-peer communication.

| Type | Default |
|------|---------|
| int | `5000` |

```toml
peer_port = 5000
```

#### `network_ip`

IP address that this node advertises to other peers. Only necessary for bootstrap configurations where other nodes connect to this node. If not specified, the node attempts to auto-detect its network IP.

| Type | Default |
|------|---------|
| string | None (auto-detect) |

```toml
network_ip = "192.168.1.100"
```

#### `bootstrap_nodes`

Array of peer nodes to contact when joining the network. Leave empty for a standalone or first node.

| Type | Default |
|------|---------|
| array of tables | `[]` (empty) |

Each entry has the following fields:

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `address` | string | ✓ | IP address of the bootstrap node |
| `port` | int | ✓ | Port of the bootstrap node |

```toml
[[bootstrap_nodes]]
address = "192.168.1.100"
port = 5000
```

#### `network_key`

Hex-encoded AES-128 key used to encrypt peer-to-peer traffic. If null or omitted, peer communication is unencrypted.

| Type | Default |
|------|---------|
| string (32 hex characters) | null |

```toml
network_key = "9f86d081884c7d659a2feaa0c55ad015"
```

Generate a key from the TUI (**Network > Configure**) or with the [`keygen`](./cli.md#keygen) command.

#### `whitelist_ips`

IP addresses this node will allow for peer communication. If configured, the node only accepts inbound DSN requests from these IPs and only sends outbound requests to these IPs.

| Type | Default |
|------|---------|
| array of strings | `[]` (allow all) |

```toml
whitelist_ips = ["192.168.1.100", "192.168.1.101"]
```

#### `whitelist_node_ids`

Peer node IDs this node will allow for communication. If configured, the node only accepts inbound DSN messages from these node IDs and only sends outbound messages to these node IDs.

| Type | Default |
|------|---------|
| array of strings | `[]` (allow all) |

```toml
whitelist_node_ids = ["bootstrap-node", "trusted-worker-1"]
```

---

## Environment Variables

These configure machine-local paths and runtime behavior. They are read
directly from the process environment and are independent of the TOML
configuration file described above.

#### `LP_APP_DIR`

Application configuration directory. Stores configs and credentials.

| Default |
|---------|
| `~/.config/language_pipes` |

```bash
export LP_APP_DIR=~/.config/language_pipes
```

Directory structure:
```
app_dir/
├── configs/     # Configuration files
└── credentials/ # Credential files
```

#### `LP_MODEL_DIR`

Model cache directory. Stores downloaded model weights.

| Default |
|---------|
| `~/.cache/language_pipes/models` |

```bash
export LP_MODEL_DIR=~/.cache/language_pipes/models
```

#### `LP_NUM_LOCAL_LAYERS`

Number of initial model layers an end model executes locally before
forwarding work to other nodes. Higher values improve prompt obfuscation by
keeping more of the early pipeline on your machine. All nodes hosting the same
end model should use the same value so that model layers are loaded correctly.

| Default |
|---------|
| `1` |

```bash
export LP_NUM_LOCAL_LAYERS=1
```

#### `LP_MAX_NODE_JOBS`

Maximum number of jobs this node will queue for a single peer node. Incoming
jobs from a node whose queue is already full are rejected.

| Default |
|---------|
| `10` |

```bash
export LP_MAX_NODE_JOBS=10
```

#### `LP_MAX_API_JOBS`

Maximum number of pending jobs allowed per API key on the
[OpenAI-compatible API](./oai.md). Requests beyond this limit are rejected
until earlier jobs for that key complete.

| Default |
|---------|
| `5` |

```bash
export LP_MAX_API_JOBS=5
```

#### `LP_8_BIT_MODE`

Load model layers in 8-bit precision using the
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library
(LLM.int8 quantization) instead of the default 16-bit floating point. This
roughly halves the memory needed for hosted layers at a small cost in output
quality and speed.

The linear projection weights of each decoder layer are quantized to int8;
the embedding, norms, and language-model head stay in `float16`.

**Note**: Requires the `bitsandbytes` package: `pip install language-pipes[quantization]` or `pip install bitsandbytes`.

| Default |
|---------|
| `false` |

```bash
export LP_8_BIT_MODE=true
```

---

## Hugging Face Authentication

#### `hf_token`

HuggingFace API token for downloading gated or private models (like Llama).
This is stored separately from node configuration files, in
`<app_dir>/globals.toml`. Language Pipes prompts for the token in the TUI when
downloading a gated model and offers to save it for future downloads.

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Note:** For gated models (like Llama), you must also accept the model's license agreement on the HuggingFace website before downloading.
