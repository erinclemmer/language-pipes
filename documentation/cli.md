# Command Line Interface

## Quick Reference

```bash
language-pipes                                  # Launch interactive TUI
language-pipes run -c config.toml               # Run headless
language-pipes config show -c config.toml       # Print effective configuration
language-pipes config validate -c config.toml   # Validate and exit
language-pipes keygen [output]                  # Generate AES encryption key
```

## Global Options

| Option | Description |
|--------|-------------|
| `-V`, `--version` | Show version and exit |
| `-h`, `--help` | Show help message and exit |

---

## Commands

### `language-pipes`

Launches the interactive TUI for creating, viewing, editing, and loading configurations.

In TUI mode the configuration file is authoritative. Environment variables and flags do not override config values. The exceptions are machine-local settings (`LP_APP_DIR`, `LP_MODEL_DIR`, `LP_HUGGINGFACE_TOKEN`) which describe the host environment rather than the node's behavior.

```bash
language-pipes
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--config FILE` | `-c` | Load configuration from TOML file | Show Main Menu |
| `--start` | | Skip startup confirmation and begin serving immediately | `false` |

---

### `run`

Start a Language Pipes server node without the TUI.

**Format:**
```bash
language-pipes run -c FILE [OPTIONS]
```

Configuration is resolved with the following precedence (highest to lowest):

1. **Environment variables** — `LP_*` prefixed variables
2. **`--set` flags** — Override individual config properties
3. **TOML config file** — Via `-c`/`--config`
4. **Defaults**

See [Configuration](./configuration.md) for all available properties, types, and defaults.

#### Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--config FILE` | `-c` | Load configuration from TOML file | Required |
| `--set KEY=VALUE` | | Override a config property by its TOML key name. Repeatable. | |
| `--layer-models MODEL...` | | Models to host (see Model Specification below) | |
| `--end-models MODEL...` | | Model IDs for which to load end models | |
| `--log-file PATH` | | Write log output to a file in addition to stdout | None |
| `--log-format FORMAT` | | `text` (default) or `json` | `text` |

The `--set` flag accepts any TOML property name and can be repeated:

```bash
language-pipes run -c config.toml \
  --set node_id=node-4 \
  --set peer_port=5001 \
  --set oai_port=9000
```

#### Model Specification

**Layer models** and **end models** use their own flags because their structure is more complex than a simple key-value pair.

Layer models are specified as comma-separated `key=value` pairs:

```bash
--set layer-models="id=MODEL,device=DEVICE,memory=GB"
```

| Key | Example |
|-----|---------|
| `id` | `Qwen/Qwen3-1.7B`, `meta-llama/Llama-3.2-1B-Instruct` |
| `device` | `cpu`, `cuda:0` |
| `memory` | `4`, `8.5` |

End models are specified as a list of model IDs:

```bash
--end-models "Qwen/Qwen3-1.7B" "meta-llama/Llama-3.2-1B-Instruct"
```

---

### `config show`

Resolve the full precedence chain (environment variables, `--set` flags, config file, defaults) and print the effective configuration as valid TOML with source annotations.

**Format:**
```bash
language-pipes config show -c FILE [OPTIONS]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config FILE` | `-c` | Configuration file to resolve | 
| `--set KEY=VALUE` | | Override a property (same as `run`) |

**Example:**

```bash
$ LP_OAI_PORT=9000 language-pipes config show -c node4.toml --set peer_port=5001

# Effective configuration
# Sources: node4.toml + environment + flags

node_id = "node-4"                    # node4.toml
oai_port = 9000                       # LP_OAI_PORT (env)
peer_port = 5001                      # --set flag
logging_level = "INFO"                # default
...
```

The output (minus comments) is valid TOML and can be piped to a file to materialize a fully resolved configuration:

```bash
language-pipes config show -c node4.toml > resolved.toml
```

---

### `config validate`

Resolve configuration the same way as `config show`, validate all values, and exit. Returns exit code `0` on success, `1` on failure. Useful for CI and pre-flight checks.

**Format:**
```bash
language-pipes config validate -c FILE [OPTIONS]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config FILE` | `-c` | Configuration file to validate |
| `--set KEY=VALUE` | | Override a property (same as `run`) |

**Example:**

```bash
$ language-pipes config validate -c node4.toml
✓ Configuration valid

$ language-pipes config validate -c broken.toml
✗ layer_models[0].device: "tpu" is not a valid PyTorch device
```

---

### `keygen`

Generate an AES encryption key for network communication.

**Format:**
```bash
language-pipes keygen [output]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `output` | Output file path | `network.key` |

**Example:**
```bash
language-pipes keygen network.key
```

---

## Examples

### Start with a config file

```bash
language-pipes run -c config.toml
```

### Override config values

```bash
language-pipes run -c config.toml \
  --set logging_level=DEBUG \
  --set oai_port=8080
```

### Start a standalone node (no config file)

```bash
language-pipes run -c minimal.toml \
  --set node_id=node-1 \
  --set oai_port=8000 \
  --layer-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" \
  --end-models "Qwen/Qwen3-1.7B"
```

### Join an existing network

```bash
language-pipes run -c config.toml \
  --set bootstrap_address=192.168.1.100
```

### Host multiple models

```bash
language-pipes run -c config.toml \
  --layer-models \
    "id=Qwen/Qwen3-1.7B,device=cpu,memory=4" \
    "id=Qwen/Qwen3-0.6B,device=cuda:0,memory=2" \
  --end-models Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B
```

### Using environment variables

```bash
export LP_NODE_ID="node-1"
export LP_OAI_PORT="8000"
export LP_LAYER_MODELS="id=Qwen/Qwen3-1.7B,device=cpu,memory=4"

language-pipes run -c config.toml
```

### Run with logging to file

```bash
language-pipes run -c node4.toml \
  --log-file node4.log \
  --status-interval 60
```

### Log to file and terminal simultaneously

```bash
language-pipes run -c node4.toml | tee node4.log
```

### Check what would run before starting

```bash
LP_OAI_PORT=9000 language-pipes config show -c node4.toml
```

### Validate config in CI

```bash
language-pipes config validate -c production.toml || exit 1
```

### Materialize a resolved config into a save file

```bash
LP_NODE_ID=node-4 language-pipes config show -c base.toml > node4.toml
```

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