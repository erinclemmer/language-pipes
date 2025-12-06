# Command Line Interface

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
language-pipes keygen [output]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `output` | Output file path | `network.key` |

```bash
language-pipes keygen network.key
```

---

### `init`

Interactively create a configuration file with guided prompts.

```bash
language-pipes init [-o FILE]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output file path | `config.toml` |

The wizard guides you through:
- Node ID configuration
- Model selection (HuggingFace ID, device, memory)
- API server settings
- Network configuration
- Advanced options

```bash
language-pipes init -o my-config.toml
```

---

### `start`

First-time setup wizard that handles everything in one command.

```bash
language-pipes start [-c FILE] [-k FILE]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-c`, `--config` | Config file path | `config.toml` |
| `-k`, `--key` | Network key file path | `network.key` |

The `start` command walks you through:

1. **Network Key Setup**
   - If first node: generates a new encryption key
   - If joining: prompts you to copy the key from the first node

2. **Configuration**
   - Runs the interactive `init` wizard if no config exists
   - Offers to reconfigure if config already exists

3. **Server Startup**
   - Prompts to start the server immediately
   - Shows the manual command if you decline

```bash
# First-time setup with defaults
language-pipes start

# Custom paths
language-pipes start -c my-config.toml -k my-network.key
```

**Recommended for new users** — this is the easiest way to get started.

---

### `serve`

Start a Language Pipes server node.

```bash
language-pipes serve [OPTIONS]
```

The `serve` command accepts configuration through three sources (in order of precedence):

1. **Command-line flags** — Override all other sources
2. **Environment variables** — `LP_*` prefixed variables
3. **TOML config file** — Via `-c`/`--config`

See [Configuration](./configuration.md) for all available options and their descriptions.

#### Common Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--config FILE` | `-c` | Load configuration from TOML file |
| `--node-id ID` | | Node identifier (required) |
| `--openai-port PORT` | | Enable OpenAI API on port |
| `--hosted-models MODEL...` | | Models to host |
| `--logging-level LEVEL` | `-l` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--bootstrap-address HOST` | | Connect to existing network |
| `--app-dir PATH` | | Application data directory (default: `~/.language-pipes`) |

Run `language-pipes serve --help` for all options.

#### Model Specification

Models are specified as comma-separated `key=value` pairs:

```bash
--hosted-models "id=MODEL,device=DEVICE,memory=GB[,load_ends=BOOL]"
```

| Key | Required | Example |
|-----|:--------:|---------|
| `id` | ✓ | `Qwen/Qwen3-1.7B`, `meta-llama/Llama-3.2-1B-Instruct` |
| `device` | ✓ | `cpu`, `cuda:0` |
| `memory` | ✓ | `4`, `8.5` |
| `load_ends` | | `true`, `false` (default) |

---

## Examples

### Start a standalone node (CLI only)

```bash
language-pipes serve \
  --node-id "node-1" \
  --openai-port 8000 \
  --hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=true"
```

### Start with config file

```bash
language-pipes serve -c config.toml
```

### Override config values

```bash
language-pipes serve -c config.toml --logging-level DEBUG --openai-port 8080
```

### Join an existing network

```bash
language-pipes serve \
  --node-id "node-2" \
  --bootstrap-address "192.168.1.100" \
  --hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4"
```

### Using environment variables

```bash
export LP_NODE_ID="node-1"
export LP_OAI_PORT="8000"
export LP_HOSTED_MODELS="id=Qwen/Qwen3-1.7B,device=cpu,memory=4"

language-pipes serve
```

### Host multiple models

```bash
language-pipes serve \
  --node-id "multi-model" \
  --openai-port 8000 \
  --hosted-models \
    "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" \
    "id=Qwen/Qwen3-0.6B,device=cuda:0,memory=2"
```

---

## Quick Start

### Easiest Method (Recommended)

```bash
language-pipes start
```

This single command handles network key generation, configuration, and server startup.

### Manual Method

```bash
# 1. Generate network key
language-pipes keygen network.key

# 2. Create configuration interactively
language-pipes init

# 3. Start server
language-pipes serve -c config.toml
```

---

## See Also

- [Configuration](./configuration.md) — TOML config file reference
- [Architecture](./architecture.md) — How Language Pipes works
- [OpenAI API](./oai.md) — API endpoint documentation
