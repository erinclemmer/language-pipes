# Command Line Interface

The Language Pipes CLI provides commands to manage distributed language model networks.

## Usage

```bash
language-pipes [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `-V`, `--version` | Show version and exit |
| `-h`, `--help` | Show help message and exit |

---

## Commands

### `keygen`

Generate an AES encryption key for network communication.

```bash
language-pipes keygen <output>
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `output` | Output file path for the AES key | `network.key` |

**Example:**
```bash
language-pipes keygen network.key
```

---

### `init`

Create a new configuration file template.

```bash
language-pipes init [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output file path | `config.toml` |

**Example:**
```bash
language-pipes init -o my-config.toml
```

---

### `serve`

Start a Language Pipes server node.

```bash
language-pipes serve [OPTIONS]
```

**Options:**

| Option | Environment Variable | Description | Default |
|--------|---------------------|-------------|---------|
| `-c`, `--config` | - | Path to TOML config file | - |
| `-l`, `--logging-level` | `LP_LOGGING_LEVEL` | Logging verbosity | `INFO` |
| `--node-id` | `LP_NODE_ID` | Unique node identifier **(Required)** | - |
| `--openai-port` | `LP_OAI_PORT` | OpenAI-compatible API port | None (disabled) |
| `--peer-port` | `LP_PEER_PORT` | Peer-to-peer network port | `5000` |
| `--job-port` | `LP_JOB_PORT` | Job receiver port | `5050` |
| `--bootstrap-address` | `LP_BOOTSTRAP_ADDRESS` | Bootstrap node address | - |
| `--bootstrap-port` | `LP_BOOTSTRAP_PORT` | Bootstrap node port | `5000` |
| `--network-key` | `LP_NETWORK_KEY` | Path to AES network key file | `network.key` |
| `--max-pipes` | `LP_MAX_PIPES` | Maximum pipes to host | `1` |
| `--model-validation` | `LP_MODEL_VALIDATION` | Validate model weight hashes | `false` |
| `--ecdsa-verification` | `LP_ECDSA_VERIFICATION` | ECDSA packet signing | `false` |
| `--hosted-models` | `LP_HOSTED_MODELS` | Models to host (see below) | - |

**Logging Levels:**
- `DEBUG` - Verbose debugging information
- `INFO` - General operational messages
- `WARNING` - Warning messages only
- `ERROR` - Error messages only

---

## Configuration Precedence

Configuration values are resolved in this order (highest to lowest priority):

1. **Command line arguments** - Explicitly passed flags
2. **Environment variables** - `LP_*` prefixed variables  
3. **TOML configuration file** - Values from `--config` file
4. **System defaults** - Built-in default values

---

## Hosted Models Format

Models are specified using comma-separated key=value pairs:

```bash
--hosted-models "id=MODEL_ID,device=DEVICE,memory=GB[,load_ends=BOOL]"
```

**Keys:**

| Key | Required | Description | Example |
|-----|----------|-------------|---------|
| `id` | Yes | HuggingFace model ID or local path | `Qwen/Qwen3-1.7B` |
| `device` | Yes | PyTorch device | `cpu`, `cuda:0`, `cuda:1` |
| `memory` | Yes | Maximum memory in GB | `4`, `8.5` |
| `load_ends` | No | Load embedding/head layers | `true`, `false` (default) |

**Examples:**
```bash
# Single model on CPU with 4GB
--hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4"

# Model on GPU with embedding/head layers
--hosted-models "id=meta-llama/Llama-3.2-1B-Instruct,device=cuda:0,memory=8,load_ends=true"

# Multiple models
--hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" "id=Qwen/Qwen3-0.6B,device=cuda:0,memory=2"
```

---

## Examples

### Start a standalone node

```bash
language-pipes serve \
  --node-id "node-1" \
  --openai-port 6000 \
  --hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=true"
```

### Start a node and connect to existing network

```bash
language-pipes serve \
  --node-id "node-2" \
  --bootstrap-address "192.168.1.100" \
  --bootstrap-port 5000 \
  --hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4"
```

### Using a config file

```bash
language-pipes serve -c config.toml
```

### Override config file values

```bash
language-pipes serve -c config.toml --logging-level DEBUG --openai-port 8080
```

### Using environment variables

```bash
export LP_NODE_ID="node-1"
export LP_OAI_PORT="6000"
export LP_HOSTED_MODELS="id=Qwen/Qwen3-1.7B,device=cpu,memory=4"

language-pipes serve
```

---

## Quick Start

1. **Generate a network key:**
   ```bash
   language-pipes keygen network.key
   ```

2. **Create a config file:**
   ```bash
   language-pipes init -o config.toml
   ```

3. **Edit `config.toml`** with your settings (see [Configuration](./configuration.md))

4. **Start the server:**
   ```bash
   language-pipes serve -c config.toml
   ```

---

## See Also

- [Configuration Reference](./configuration.md) - Detailed TOML configuration options
- [Architecture](./architecture.md) - How Language Pipes works internally
- [OpenAI API](./oai.md) - Using the OpenAI-compatible endpoint
