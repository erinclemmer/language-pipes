---
title: Command Line Interface
description: The Language Pipes CLI — launching the TUI, running saved TOML configurations headlessly, and every available flag.
---

## Quick Reference

```bash
language-pipes                              # Launch interactive TUI
language-pipes -c config.toml               # Launch TUI with a config preloaded
language-pipes -c config.toml --start       # Launch TUI and start serving immediately
language-pipes -c config.toml run           # Run headless from a config file
language-pipes -c config.toml config        # Print the configuration
```

> **Argument order matters.** `-c`/`--config`, `--start`, `-v`, and `-h` are
> options on the top-level command and must appear **before** the subcommand
> (`run`, `config`, `keygen`). For example, `language-pipes run -c config.toml`
> fails — use `language-pipes -c config.toml run`.

## Global Options

These are parsed by the top-level `language-pipes` command and apply to every
subcommand. They must be given before the subcommand name.

| Option | Description | Default |
|--------|-------------|---------|
| `-h`, `--help` | Show help message and exit | |
| `-v`, `--version` | Print the version and exit | |
| `-c FILE`, `--config FILE` | Configuration to load (see below) | Show Main Menu |
| `--start` | Skip the startup confirmation and begin serving immediately | `false` |

### How `--config` is resolved

The value passed to `-c`/`--config` is interpreted as follows:

- If it contains `.toml`, it is treated as a path to a TOML file.
- Otherwise it is treated as the name of a saved configuration and resolved to
  `<app_dir>/configs/<name>.toml`.

If the resolved file does not exist, the command exits immediately with:

```
ERROR: <value> is not a valid path or saved configuration
```

See [Configuration](./configuration.md) for all available properties, types,
and defaults.

---

## Commands

### `language-pipes` (no subcommand)

Launches the interactive TUI for creating, viewing, editing, and loading
configurations.

```bash
language-pipes                          # Open the main menu
language-pipes -c config.toml           # Open with a configuration preloaded
language-pipes -c config.toml --start   # Open and begin serving immediately
```

- `-c`/`--config` preloads a configuration instead of showing the main menu.
- `--start` begins serving all configured services without waiting for
  confirmation.

In TUI mode the configuration file is authoritative. Environment variables and
flags do not override config values.

---

### `run`

Start a Language Pipes server node without the TUI, streaming output to stdout.

**Format:**
```bash
language-pipes -c FILE run
```

A configuration is required. If `-c`/`--config` is not provided, the command
exits with:

```
ERROR: --config param required
```

The configuration file is resolved the same way as the global `--config` option
(a `.toml` path, or a saved configuration name under `<app_dir>/configs/`).

```bash
language-pipes -c config.toml run
language-pipes -c node4 run            # resolves <app_dir>/configs/node4.toml
```

---

### `config`

Resolve a configuration file and print its settings as a human-readable report
(ports, API keys, layer models, end models, and network settings).

**Format:**
```bash
language-pipes -c FILE config
```

A configuration is required (`-c`/`--config`), resolved the same way as `run`.

**Example:**

```bash
$ language-pipes -c node4.toml config
============================================================
--- Configuration Settings ---
============================================================

Job Port: 8000
API Keys:
- None

Layer Models:
- None

End Models:
- None

============================================================
  DSNode Configuration Details
============================================================

--- Node Settings ---
  Node ID:           node-4
  ...
```

> The output is a formatted report intended for inspection. It is **not** valid
> TOML and is not designed to be piped back into a configuration file.

---

## Examples

#### Launch the TUI with a saved configuration

```bash
language-pipes -c node4
```

#### Run a node headless from a config file

```bash
language-pipes -c config.toml run
```

#### Run a node headless and log to a file as well as the terminal

```bash
language-pipes -c node4.toml run | tee node4.log
```

#### Inspect the configuration

```bash
language-pipes -c node4.toml config
```