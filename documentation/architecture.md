# Architecture Overview

Language Pipes distributes large language model inference across multiple machines by splitting models into segments and coordinating computation between nodes. This document explains how the system works internally, its key components, and the flow of inference requests.

## Core Concept: Distributed Model Segments

Traditional language model inference loads the entire model into memory on a single machine. Language Pipes instead splits models into **segments** that can be distributed across multiple nodes:

```
Traditional Setup:
[Full Model] → Single Machine (32GB+ RAM required)

Language Pipes:
[Embedding] → Node A (2GB RAM)
[Layers 0-15] → Node B (16GB RAM) 
[Layers 16-31] → Node C (16GB RAM)
[Head/Norm] → Node D (2GB RAM)
```

Each node only needs enough memory for its assigned segments, making large models accessible to networks of smaller machines.

## System Architecture
This architecture enables democratized access to large language models by distributing computational and memory requirements across networks of smaller machines, while maintaining reasonable performance characteristics for many use cases.

### High-Level Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Node A      │    │     Node B      │    │     Node C      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ JobManager  │ │    │ │ JobManager  │ │    │ │ JobManager  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │JobReceiver  │◄┼────┼►│JobReceiver  │◄┼────┼►│JobReceiver  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ LlmModel    │ │    │ │ LlmModel    │ │    │ │ LlmModel    │ │
│ │ (Embedding) │ │    │ │ (Layers)    │ │    │ │ (Head/Norm) │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. JobManager
- **Purpose**: Orchestrates job distribution and completion across the network
- **Key Functions**:
  - Manages local model segments
  - Routes jobs to appropriate nodes
  - Handles job completion and error recovery
  - Maintains network-wide view of available model segments

#### 2. JobReceiver  
- **Purpose**: Receives and processes computation jobs from other nodes
- **Key Functions**:
  - Accepts incoming HTTPS/HTTP requests containing jobs
  - Validates job signatures for security
  - Queues jobs for processing
  - Sends responses back to requesting nodes

#### 3. LlmModel
- **Purpose**: Represents a model segment hosted on a local node
- **Key Properties**:
  - Model segments (embedding, specific layers, head/norm)
  - Device allocation (CPU/GPU)
  - Memory usage tracking
  - Computation capabilities

#### 4. Pipe
- **Purpose**: Represents a complete model distributed across multiple nodes
- **Key Functions**:
  - Tracks which nodes host which model segments
  - Routes jobs through the correct sequence of nodes
  - Validates model completeness before accepting jobs

## Inference Flow

### Job Processing Pipeline

Every inference request flows through five distinct computation steps:

```
1. TOKENIZE  →  2. EMBED  →  3. LAYER  →  4. NORM  →  5. HEAD
   ┌─────────┐    ┌───────┐    ┌───────┐    ┌──────┐    ┌──────┐
   │ Text to │    │ Token │    │Process│    │ RMS  │    │Output│
   │ tokens  │    │ embed │    │layers │    │ Norm │    │token │
   └─────────┘    └───────┘    └───────┘    └──────┘    └──────┘
```

#### Step 1: TOKENIZE
- **Location**: Any node with the model's tokenizer
- **Process**: Convert input text to numerical token IDs
- **Output**: Array of token IDs representing the input

#### Step 2: EMBED  
- **Location**: Node hosting the embedding layer
- **Process**: Convert token IDs to dense vector representations
- **Output**: High-dimensional embeddings for each token

#### Step 3: LAYER
- **Location**: Nodes hosting transformer layers (may span multiple nodes)
- **Process**: Apply attention and feed-forward transformations
- **Output**: Transformed hidden states
- **Special Handling**: Jobs may visit multiple nodes sequentially as they progress through layer ranges

#### Step 4: NORM
- **Location**: Node hosting the RMS normalization layer
- **Process**: Apply final normalization to hidden states
- **Output**: Normalized hidden states ready for output projection

#### Step 5: HEAD
- **Location**: Node hosting the language modeling head
- **Process**: Project hidden states to vocabulary logits and select next token
- **Output**: Selected token ID for the response

### Multi-Token Generation

For generating complete responses (not just single tokens), the system repeats the inference pipeline:

```
Token 1: TOKENIZE → EMBED → LAYER → NORM → HEAD → Token ID
Token 2: Use Token 1 output → EMBED → LAYER → NORM → HEAD → Token ID  
Token 3: Use Token 1+2 output → EMBED → LAYER → NORM → HEAD → Token ID
...continue until stop token or max length
```

## Network Coordination

### Peer Discovery and State Management

Language Pipes uses a distributed state network for peer discovery and coordination:

- **Node Registration**: Each node announces its available model segments
- **State Synchronization**: Network maintains consistent view of all available segments
- **Dynamic Membership**: Nodes can join/leave without manual reconfiguration
- **Health Monitoring**: Failed nodes are detected and routes updated automatically

### Model Segment Allocation

When a node starts, it automatically determines which model segments to load based on:

1. **Available Memory**: Calculates how much of the model fits in allocated memory
2. **Network Gaps**: Identifies missing segments in existing network pipes  
3. **Load Balancing**: Avoids overloading specific nodes or network links

## Privacy Architecture

### The End Model: Keeping Prompts Local

A key design principle of Language Pipes is that **your prompt text never needs to leave your computer**. This is achieved through the **End Model** architecture.

The End Model consists of three components grouped together:
- **Embedding Layer** — Converts text tokens into numerical vectors
- **RMS Normalization** — Final normalization before output
- **Output Head** — Converts hidden states back to token probabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                     END MODEL NODE                              │
│                                                                 │
│  "Hello AI" ──► Tokenizer ──► [15496, 9552] ──► Embedding      │
│                                                   ↓             │
│                                    Hidden State: [0.23, -0.14]  │
└───────────────────────────────────────┬─────────────────────────┘
                                        │
                    Only hidden states  │  (numerical tensors)
                    leave your machine  │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER NODES                                 │
│                                                                 │
│  Receive tensors ──► Process through layers ──► Return tensors │
│  [0.23, -0.14, ...]       (no text access)      [0.87, 0.12]   │
└─────────────────────────────────────────────────────────────────┘
```

### What Each Node Sees

| Node Type | Sees Raw Text | Sees Token IDs | Sees Hidden States |
|-----------|:-------------:|:--------------:|:------------------:|
| End Model | ✓ | ✓ | ✓ |
| Layer Node | ✗ | ✗ | ✓ |

**Layer nodes cannot reconstruct your prompts.** Without the embedding layer weights and tokenizer vocabulary, the hidden state tensors are meaningless numerical arrays.

### Privacy Deployment Pattern

For maximum privacy, host the End Model yourself:

```toml
# Your machine - keeps prompts private
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
load_ends = true    # ← You control the End Model
max_memory = 2

# Friend's GPU - contributes compute, never sees prompts  
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
load_ends = false   # ← Only processes tensors
max_memory = 8
```

---

## Security Architecture

### Certificate-Based Trust

- **Node Identity**: Each node has a unique certificate for identification
- **Message Signing**: All job messages are cryptographically signed
- **Verification**: Receiving nodes verify sender identity before processing

### Encrypted Communication

- **Transport Security**: HTTPS used for all inter-node communication
- **Network Keys**: Shared network keys for joining trusted networks
- **Data Protection**: Job payloads encrypted during transmission

## Integration Points

### OpenAI-Compatible API

The system exposes a standard OpenAI-compatible endpoint that allows seamless integration with existing applications expecting OpenAI API format:  

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama-7b",
  "messages": [...],
  "max_completion_tokens": 100
}
```
See [OpenAI's documentation](https://platform.openai.com/docs/api-reference/chat) for a more complete explanation.


### External Dependencies

- **[HuggingFace Integration](https://huggingface.co)**: Automatic model download and tokenizer loading
- **[PyTorch Backend](https://pytorch.org)**: Model computation and tensor operations  
- **[Distributed State Network](https://github.com/erinclemmer/distributed_state_network)**: Peer discovery and coordination
