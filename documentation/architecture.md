# Architecture Overview

Language Pipes distributes transformer inference across multiple machines by splitting a model into **layer segments** that are coordinated over a peer-to-peer control plane. This document describes how the current implementation works, from node startup through job execution.

## High-Level Runtime Components

Each node runs a single `LanguagePipes` instance that wires together networking, model hosting, and job execution:

- **DSNodeServer (distributed_state_network)**: peer discovery and shared state storage (e.g., model metadata, job ports).
- **RouterPipes**: aggregates the shared model metadata into logical pipes.
- **ModelManager**: loads model segments (layers) and optional model ends (embedding/norm/head) based on memory limits.
- **PipeManager**: merges local segments with remote metadata to create runnable `Pipe` objects.
- **JobFactory**: creates new jobs from API requests and injects them into the job pipeline.
- **JobReceiver + JobServer**: HTTP server that receives serialized job payloads and runs the job processor FSM.
- **OAIHttpServer**: optional OpenAI-compatible API that converts requests into jobs.

```
Client ──► OAIHttpServer ──► JobFactory ──► JobReceiver ──► JobProcessor FSM
                    ▲                                  │
                    └──────── results/stream updates ──┘
```

## Model Metadata and Pipe Construction

### MetaModel and MetaPipe (control plane)

Each loaded segment publishes a `MetaModel` record (process ID, layer range, model ID, pipe ID) to the distributed state network. `RouterPipes` collects every node’s `MetaModel` list and aggregates them into `MetaPipe` objects keyed by `pipe_id`.

A `MetaPipe` is **complete** when its segments cover layer 0 through `num_layers - 1` without gaps and each segment is marked `loaded`.

### Pipe (data plane)

A `Pipe` is built from a `MetaPipe` plus any locally hosted `LlmModel` segments. The resulting `Pipe` contains:

- **Local segments** (real `LlmModel` instances).
- **Remote segments** (virtual `LlmModel` placeholders from metadata).

The `Pipe` exposes helpers for:

- `get_layer(start_layer)`: find the segment responsible for the current layer index.
- `send_job(...)`: send a serialized job to another node’s job server (HTTP POST).
- `is_complete()`: check whether the segment list covers the full layer range.

## Model Hosting and Placement

`ModelManager` hosts model segments based on configuration:

- **Hosted models** are declared with `id`, `device`, `max_memory`, and `load_ends`.
- For each hosted model, the manager estimates how many layers fit in memory using `ComputedData.avg_layer_size`.
- It tries to **fill gaps** in existing pipes first, then creates a new pipe if `max_pipes` allows it.
- When `load_ends` is enabled, the node also loads the **EndModel** (embedding, RMS norm, output head, tokenizer).
- If `model_validation` is enabled, computed hashes must match the network pipe’s hashes before loading.

## Inference Flow (Jobs)

### Request Entry

1. **API request** hits `OAIHttpServer` at `/v1/chat/completions`.
2. The handler constructs a `ChatCompletionRequest` and calls `JobFactory.start_job(...)`.
3. `JobFactory` finds an available `Pipe` for the requested model, creates a `Job`, and sends the initial `NetworkJob` to the **origin node** (usually itself).

### Job Processing FSM

Each job is processed by `JobProcessor`, a finite-state machine driven by the job's `ComputeStep`. See [JobProcessor State Machine](./job-processor.md) for detailed state transition documentation.

```
TOKENIZE → EMBED → LAYER → NORM → HEAD → (repeat for next token)
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

`JobReceiver` validates the hash before enqueuing a job; if validation fails, it requests a restart by sending the job back to the origin for re-embedding.

## Distributed State and Coordination

Language Pipes uses the **distributed_state_network** control plane for peer discovery and shared metadata:

- Nodes advertise **model segments** in a `models` key.
- Nodes advertise their **job receiver port** in `job_port`.
- `RouterPipes` aggregates these entries across all peers to build the available pipes.

Job payloads themselves are sent over **HTTP** to the advertised job port. The control plane can be configured with network keys and optional ECDSA verification via `DSNodeConfig`, while job payloads use hash validation at the application layer.

## Privacy Architecture (End Model)

Only the node hosting the **EndModel** can see raw text:

- The **EndModel** handles tokenization, embedding, normalization, and head projection.
- All other nodes only receive **hidden-state tensors** and cannot decode text without the tokenizer and embedding/head weights.

For privacy-sensitive deployments, keep `load_ends = true` on your own machine and let other nodes host only layer segments.

## Observability and Diagnostics

- **Job timing**: `NetworkJob.times` tracks per-hop processing and network round-trip timing.
- **Prefill logging**: chunked prefill logs per-chunk timing and throughput.
- **Optional debug output**: `print_job_data` dumps job configuration when enabled.
