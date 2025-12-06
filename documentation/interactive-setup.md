# Interactive Setup Guide

This document explains the interactive configuration wizard in Language Pipes, which provides a guided way to create and manage configuration files.

For configuration file reference, see [Configuration](./configuration.md).  
For command-line options, see [CLI Reference](./cli.md).

---

## Overview

Language Pipes provides two levels of interactive setup:

1. **Start Wizard** — A main menu for managing multiple configurations
2. **Configuration Wizard** — A step-by-step guide for creating individual configs

---

## Start Wizard

The start wizard is the primary entry point for new users. Launch it with:

```bash
language-pipes start
```

Or simply:

```bash
language-pipes
```

### Main Menu

The wizard presents a menu with four options:

```
Main Menu
[0] View Config
[1] Load Config
[2] Create Config
[3] Delete Config
Select number of choice: 
```

| Option | Description |
|--------|-------------|
| **View Config** | Display the contents of an existing configuration file |
| **Load Config** | Select and start the server with an existing configuration |
| **Create Config** | Launch the configuration wizard to create a new config |
| **Delete Config** | Remove an existing configuration file |

### Application Data Directory

On first run, the wizard prompts for the application data directory:

```
Where should we store application data? [~/.language-pipes]: 
```

This directory stores:
- `configs/` — Configuration files created by the wizard
- `credentials/` — Node credentials
- `models/` — Downloaded model weights

---

## Configuration Wizard

The configuration wizard guides you through creating a complete configuration file. Launch it directly with:

```bash
language-pipes init
```

Or select "Create Config" from the start wizard.

### Sections

The wizard is organized into logical sections:

1. [Required Settings](#required-settings) — Node identity
2. [Model Configuration](#model-configuration) — Which models to host
3. [API Server](#api-server) — OpenAI-compatible endpoint
4. [Network Configuration](#network-configuration) — Peer-to-peer settings
5. [Advanced Options](#advanced-options) — Security and limits

---

### Required Settings

#### Node ID

```
The node ID is a unique name that identifies this computer on the
Language Pipes network. Other nodes will use this to route jobs.

Node ID [your-hostname]: 
```

The default is your system hostname. Choose a descriptive name if running multiple nodes.

---

### Model Configuration

This section configures which LLM models your node will host.

#### Model ID

```
Enter the HuggingFace model ID. This is the identifier used on
huggingface.co (e.g., 'Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.2-1B-Instruct').

Model ID [Qwen/Qwen3-1.7B]: 
```

Use any model ID from [HuggingFace](https://huggingface.co/models). The model will be downloaded automatically on first use.

#### Device

```
Select the compute device for this model. Use 'cpu' for CPU-only,
or 'cuda:0', 'cuda:1', etc. for specific GPUs.

Device [cpu]: 
```

| Value | Description |
|-------|-------------|
| `cpu` | Run on CPU (slower, but always available) |
| `cuda:0` | First NVIDIA GPU |
| `cuda:1` | Second NVIDIA GPU |
| `cuda:N` | Nth GPU (zero-indexed) |

#### Max Memory

```
Specify the maximum RAM/VRAM (in GB) this node should use for the model.
The model layers will be loaded until this limit is reached.

Max memory (GB) [4]: 
```

The system loads model layers until this limit is reached. Remaining layers are distributed to other nodes in the network.

#### Load Ends (End Model)

```
The 'ends' of a model are the embedding layer (input) and the output head.
At least one node in the network needs these loaded to process requests.
If this is the only node or the first/last in a chain, enable this.

Load embedding/output layers [Y/n]: 
```

**Privacy Note:** The node with "ends" enabled is the only node that sees your actual prompts and responses. Other nodes only process numerical tensors and cannot read your conversations.

| Scenario | Setting | Privacy Implication |
|----------|---------|---------------------|
| Single node / standalone | Enable (`Y`) | You control all data |
| **You want prompt privacy** | Enable (`Y`) | Your prompts stay on your machine |
| Contributing compute only | Disable (`n`) | You never see users' prompts |
| Middle node in chain | Disable (`n`) | Only processes hidden states |

**Recommended:** If privacy matters, always enable this on your own machine and let others contribute compute with it disabled.

#### Multiple Models

After configuring each model, you can add more:

```
Add another model? [y/N]: 
```

---

### API Server

```
Language Pipes can expose an OpenAI-compatible HTTP API, allowing you to
use standard OpenAI client libraries (Python, JavaScript, curl, etc.)
to interact with your distributed model.

Enable OpenAI-compatible API server? [Y/n]: 
```

If enabled:

```
Choose a port for the API server. Clients will connect to
http://<this-machine>:<port>/v1/chat/completions

API port [8000]: 
```

See [OpenAI API Reference](./oai.md) for endpoint documentation.

---

### Network Configuration

#### First Node vs. Joining

```
Language Pipes uses a peer-to-peer network to coordinate between nodes.
The first node starts fresh; additional nodes connect to an existing node.

Is this the first node in the network? [Y/n]: 
```

If joining an existing network:

```
Enter the IP address of an existing node on the network.
This node will connect to it to join the distributed network.

Bootstrap node IP: 

Enter the peer port of the bootstrap node (default is 5000).

Bootstrap node port [5000]: 
```

#### Ports

```
The peer port is used for network coordination and discovery.
Other nodes will connect to this port to join the network.

Peer port [5000]: 

The job port is used for transferring computation data between nodes
during model inference (hidden states, embeddings, etc.).

Job port [5050]: 
```

| Port | Purpose | Default |
|------|---------|---------|
| Peer port | Network coordination, node discovery | 5000 |
| Job port | Inference data transfer | 5050 |

#### Network Encryption

```
The network key is an AES encryption key shared by all nodes.
It encrypts communication and prevents unauthorized access.
There will be no encryption between nodes if the default value is selected.

Encrypt network traffic [y/N]: 
```

If enabled:

```
Network key [Generate new key]: 
Generated new key: a1b2c3d4e5f6...
Note: Save this key somewhere and supply it to other nodes on the network
```

All nodes in a network must share the same key. Save the generated key and provide it when configuring additional nodes.

---

### Advanced Options

```
Advanced options include logging verbosity, security features, and limits.

Configure advanced options? [y/N]: 
```

#### Logging Level

```
Controls how much information is printed to the console.
DEBUG shows everything, ERROR shows only critical issues.

Logging level (DEBUG/INFO/WARNING/ERROR) [INFO]: 
```

#### Max Pipes

```
Limits how many model 'pipes' (distributed model instances)
this node will participate in simultaneously.

Max pipes [1]: 
```

#### Model Validation

```
When enabled, nodes verify that model weight hashes match
to ensure all nodes are running the exact same model.

Validate model hashes? [y/N]: 
```

Enable this for production deployments to prevent model mismatches.

#### ECDSA Verification

```
ECDSA verification signs each job packet cryptographically,
ensuring jobs only come from authorized nodes in the pipe.

Enable ECDSA signing? [y/N]: 
```

Adds cryptographic verification overhead but prevents unauthorized job injection.

---

### Configuration Summary

After completing all sections, the wizard displays a summary:

```
==================================================
  Configuration Summary
==================================================

node_id = "my-node"
oai_port = 8000
peer_port = 5000
job_port = 5050

[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4.0
load_ends = true
```

The wizard then saves the configuration to the specified path.

---

## Input Types

The wizard uses several input methods:

### Text Input

```
Prompt [default]: your-input
```

Press Enter to accept the default value, or type a new value.

### Boolean Input

```
Prompt [Y/n]: 
```

| Input | Accepted Values |
|-------|-----------------|
| Yes | `y`, `yes`, `true`, `1` |
| No | `n`, `no`, `false`, `0` |

The capitalized letter indicates the default.

### Number Input

```
Prompt [5000]: 
```

Enter an integer value or press Enter for the default.

### Choice Selection

```
Prompt (DEBUG/INFO/WARNING/ERROR) [INFO]: 
```

Enter one of the listed values exactly.

### Numbered Menu

```
Main Menu
[0] Option A
[1] Option B
[2] Option C
Select number of choice: 
```

Enter the number corresponding to your choice.

---

## File Locations

| Location | Description |
|----------|-------------|
| `~/.language-pipes/configs/` | Configurations created via start wizard |
| `./config.toml` | Default output for `language-pipes init` |
| Custom path | Specify with `language-pipes init -o path/to/config.toml` |

---

## Examples

### First Node Setup

```bash
$ language-pipes start

Where should we store application data? [~/.language-pipes]: 

Main Menu
[0] View Config
[1] Load Config
[2] Create Config
[3] Delete Config
Select number of choice: 2

Name of new configuration: production

==================================================
  Configuration Setup
==================================================

--- Required Settings ---

Node ID [workstation]: primary-node

--- Model Configuration ---

Model ID [Qwen/Qwen3-1.7B]: meta-llama/Llama-3.2-1B-Instruct
Device [cpu]: cuda:0
Max memory (GB) [4]: 8
Load embedding/output layers [Y/n]: y
Add another model? [y/N]: n

--- API Server ---

Enable OpenAI-compatible API server? [Y/n]: y
API port [8000]: 8000

--- Network Configuration ---

Is this the first node in the network? [Y/n]: y
Peer port [5000]: 5000
Job port [5050]: 5050
Encrypt network traffic [y/N]: y
Network key [Generate new key]: 
Generated new key: 7f3a9b2c...
Note: Save this key somewhere and supply it to other nodes on the network

--- Advanced Options ---

Configure advanced options? [y/N]: n

✓ Configuration saved
```

### Joining an Existing Network

```bash
$ language-pipes init -o node2.toml

# ... (earlier sections) ...

--- Network Configuration ---

Is this the first node in the network? [Y/n]: n
Bootstrap node IP: 192.168.1.100
Bootstrap node port [5000]: 5000
Peer port [5000]: 5001
Job port [5050]: 5051
Encrypt network traffic [y/N]: y
Network key [Generate new key]: 7f3a9b2c...

# ... (remaining sections) ...
```

---

## See Also

- [Configuration Reference](./configuration.md) — TOML config file format
- [CLI Reference](./cli.md) — All command-line options
- [Architecture](./architecture.md) — How Language Pipes works
- [OpenAI API](./oai.md) — API endpoint documentation
