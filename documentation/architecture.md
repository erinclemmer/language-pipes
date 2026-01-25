# Architecture Overview

Language Pipes distributes transformer inference across multiple machines by splitting a model into **layer segments** that are coordinated over a peer-to-peer control plane. This document describes how the protocol works, from node startup through job execution.

## High-Level Runtime Components

Each node runs a single `LanguagePipes` instance that wires together networking, model hosting, and job execution:

- **Router (distributed_state_network by default)**: peer discovery and shared state storage (e.g., model metadata, job ports).
- **RouterPipes**: aggregates the shared model metadata into logical pipes
- **ModelManager**: loads model segments (layers) and optional model ends (embedding/norm/head) based on memory limits.
- **PipeManager**: merges local segments with remote metadata to create runnable `Pipe` objects.
- **JobFactory**: creates new jobs from API requests and injects them into the job pipeline.
- **JobReceiver**: HTTP server that receives serialized job payloads and runs the job processor FSM.
- **JobProcessor**: State machine for jobs on the network. Jobs received by job receiver will run in this class.
- **OAIHttpServer**: optional OpenAI-compatible API that converts requests into jobs.

```
Client ──► OAIHttpServer ──► JobFactory ──► JobReceiver ──► JobProcessor
                    ▲                                             │
                    └──────── results/stream updates ─────────────┘
```

## Model Hosting and Placement

`ModelManager` hosts model segments based on configuration:

- **Hosted models** are declared with `id`, `device`, `max_memory`, and `load_ends`.
- For each hosted model, the manager estimates how many layers fit in the memory budget.
- It tries to **fill gaps** in existing pipes first, then creates a new pipe if `max_pipes` allows it.
- When `load_ends` is enabled, the node also loads the **EndModel** (embedding, RMS norm, output head, tokenizer).
- If `model_validation` is enabled, computed hashes must match the network pipe’s hashes before loading.

## Model Metadata and Pipe Construction

### MetaModel and MetaPipe

Each loaded segment publishes a `MetaModel` record to the distributed database. This record contains metadata associated with a model that is loaded in memory. This includes the node ID of the node loading it, model ID that is loaded, the process ID of the model, the pipe ID that the model is associated with, the range of layers loaded, etc. `RouterPipes` collects every node’s `MetaModel` list and aggregates them into `MetaPipe` objects.

A `MetaPipe` is a logical representation of a full or partial model on the network. Each pipe is for a specific model that is denoted by the model ID. If two nodes join the network that are trying to load the same model ID they will try to load layers that the other node is missing. 

For example if node A joins the network first for a particular model ID and loads layer 1-5 and node B joins the network afterward it will load layers 6-10 (or however many fit into its memory budget).

### Pipe

A `Pipe` is built from a `MetaPipe` plus any locally hosted `LlmModel` segments. The resulting `Pipe` contains:

- **Local segments** (real `LlmModel` instances).
- **Remote segments** (virtual `LlmModel` placeholders from metadata).

The `Pipe` exposes helpers for:

- `get_layer(start_layer)`: find the segment responsible for the current layer index.
- `send_job(...)`: send a serialized job to another node’s job server using the router.
- `is_complete()`: check whether the segment list covers the full layer range.

## Inference Flow (Jobs)

### Request Entry

1. **API request** hits `OAIHttpServer` at `/v1/chat/completions`.
2. The handler constructs a `ChatCompletionRequest` and calls `JobFactory.start_job(...)`.
3. `JobFactory` finds an available `Pipe` for the requested model, creates a `Job`, and sends the initial `NetworkJob` to the `JobProcessor`.

### Job Processing FSM

Each job is processed by `JobProcessor`, a finite-state machine driven by the job's `ComputeStep`. See [JobProcessor State Machine](./job-processor.md) for detailed state transition documentation.

```
TOKENIZE → EMBED → LAYER → NORM → HEAD → (back to EMBED for next token)
```

Key behaviors:

- **TOKENIZE/EMBED (origin only)**
  - The origin node must have the **EndModel** loaded.
  - `EndModel.tokenize` builds the prompt using the tokenizer’s chat template and encodes IDs.
  - `EndModel.compute_embed` produces the initial hidden state and attaches it to `JobData`.
  - Prefill is chunked using `prefill_chunk_size` to reduce latency for large prompts; the FSM sends intermediate updates after each chunk.

- **LAYER (distributed)**
  - The current layer index (`job.current_layer`) determines which segment should run next.
  - If the segment is local, `LlmModel.process_job` runs through its range and updates the hidden state.
  - If the segment is remote, the job is serialized and sent to the next node via HTTP.
  - Each segment sets `current_layer = end_layer + 1` so the next hop starts at the correct boundary.

- **NORM/HEAD (origin only)**
  - Once layers are complete, the job returns to the origin.
  - The origin runs `compute_norm` and `compute_head`, samples the next token, and updates job status.
  - If generation is complete, the result is decoded and the job is marked complete; otherwise the FSM loops back to EMBED for the next token.

### Network Job Serialization

`NetworkJob` payloads include:

- Job metadata (IDs, pipe ID, compute step, current layer).
- `JobData` tensors: hidden state, position IDs, attention masks, and cache position.
- A SHA-256 hash of the job state for integrity.

**Note:** The `NetworkJob` object does not contain the original prompt.

`JobReceiver` validates the hash before enqueuing a job; if validation fails, it requests a restart by sending the job back to the origin for re-embedding and the current token is restarted.

## Network Agnostic Architecture
Language Pipes is designed to be network agnostic except for a few assumptions. The network layer is expected to handle peer discovery, encryption, and data transfer.

**Note:** To be more flexible we denote a "pipe system" here as a network or network partition that is expected to put pipe parts together. The default implementation (DistributedStateNetwork) uses one pool  for all nodes but that does not have to be the case as long as the router supports everything that DSN suports.

These are the assumptions about the router that Language Pipes makes:
- The network hosts a distributed database where each node can write data about themselves to the network.
- All nodes in a pipe system can see each other's database entries.
- The network operates by unique node IDs and each node can see all other node IDs in the pipe system.
- The network supports a way of sending data from one node to another using a node ID.

See `src/language_pipes/network_protocol.py` for the exact network protocol specification. If a network can be made that satisifies this interface then Language Pipes can be made to run on top of it.

## Privacy Architecture (End Model)

Only the node hosting the **EndModel** can see raw text:

- The **EndModel** handles tokenization, embedding, normalization, and head projection.
- All other nodes only receive **hidden-state tensors** which do not have prompt data in them.

For privacy-sensitive deployments, keep `load_ends = true` on your own machine and let other nodes host only layer segments.

For more information how the architecture ensures privacy see [the privacy documentation](./privacy.md).
