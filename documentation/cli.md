# Command Line Interface

### CLI Wizard

This provides an easy to use CLI interface for managing configurations.

```bash
language-pipes
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
**Format:**
```bash
language-pipes keygen [output]
```
**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `output` | Output file path | `network.key` |

**Example:**
```bash
language-pipes keygen network.key
```

---

### `init`

Interactively create a configuration file with guided prompts.

**Format:**
```bash
language-pipes init [FILE]
```

**Arguments:**
| Option | Description | Default |
|--------|-------------|---------|
| `output` | Output file path | `config.toml` |

**Example:**
```bash
language-pipes init my-config.toml
```

---

### `serve`

Start a Language Pipes server node.

**Format:**
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
| `--app-dir PATH` | | Application config directory (default: `~/.config/language_pipes`) |
| `--model-dir PATH` | | Model cache directory (default: `~/.cache/language_pipes/models`) |
| `--print-times` | | Print timing info for layer computations and network transfers |

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

## See Also

- [Configuration](./configuration.md) — TOML config file reference
- [Architecture](./architecture.md) — How Language Pipes works
- [JobProcessor State Machine](./job-processor.md) — Job processing FSM details
- [OpenAI API](./oai.md) — API endpoint documentation
