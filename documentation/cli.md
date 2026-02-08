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

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--config FILE` | `-c` | Load configuration from TOML file | None |
| `--node-id ID` | | Node identifier (required) | Required |
| `--openai-port PORT` | | Enable OpenAI API on port | None |
| `--layer-models MODEL...` | | Models to host (layers) | Empty|
| `--end-models MODEL...` | | Model IDs for which to load end models | Empty | 
| `--num-local-layers N` | | Number of local layers to run on your machine (higher values improve prompt obfuscation) | 1 |
| `--logging-level LEVEL` | `-l` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | INFO |
| `--bootstrap-address HOST` | | Connect to existing network | None |
| `--app-dir PATH` | | Application config directory | `~/.config/language_pipes` |
| `--model-dir PATH` | | Model cache directory | `~/.cache/language_pipes/models` |
| `--print-times` | | Print timing info for layer computations and network transfers | False|

Run `language-pipes serve --help` for all options.

#### Model Specification

**Layer models** are specified as comma-separated `key=value` pairs:

```bash
--layer-models "id=MODEL,device=DEVICE,memory=GB"
```

| Key | Example |
|-----|---------|
| `id` | `Qwen/Qwen3-1.7B`, `meta-llama/Llama-3.2-1B-Instruct` |
| `device` | `cpu`, `cuda:0` |
| `memory` | `4`, `8.5` |

**End models** are specified as a list of model IDs:

```bash
--end-models "Qwen/Qwen3-1.7B" "meta-llama/Llama-3.2-1B-Instruct"
```

---

## Examples

### Start a standalone node (CLI only)

```bash
language-pipes serve \
  --node-id "node-1" \
  --openai-port 8000 \
  --layer-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" \
  --end-models "Qwen/Qwen3-1.7B"
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
  --layer-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4"
```

### Using environment variables

```bash
export LP_NODE_ID="node-1"
export LP_OAI_PORT="8000"
export LP_LAYER_MODELS="id=Qwen/Qwen3-1.7B,device=cpu,memory=4"

language-pipes serve
```

### Host multiple models

```bash
language-pipes serve \
  --node-id "multi-model" \
  --openai-port 8000 \
  --layer-models \
    "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" \
    "id=Qwen/Qwen3-0.6B,device=cuda:0,memory=2" \
  --end-models Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B
```

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [Open AI Compatable API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [The default peer to peer implementation](./distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./llm-layer-collector.md)
