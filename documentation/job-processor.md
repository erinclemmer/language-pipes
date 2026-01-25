# JobProcessor State Machine

The `JobProcessor` class implements a finite state machine (FSM) that orchestrates job execution across the distributed inference pipeline. This document describes each state, the conditions for transitions, and how the FSM integrates with the broader Language Pipes architecture.


## Overview

When a job arrives at a node (via `JobReceiver`), it is processed by a `JobProcessor` instance. The processor validates the job context, routes computation through local or remote model segments, and handles job completion or handoff.


## States

### `VALIDATING`

**Purpose:** Validate the job context before processing begins.

The FSM starts in this state for every job. It checks that all required resources are available:

- The job object exists
- The pipe is available and complete (all layer segments are ready)
- For `HEAD` compute steps: the origin node matches and the end model is loaded

**Transitions:**

| Condition | Next State |
|-----------|------------|
| Job is missing | `DONE` |
| Pipe is unavailable or incomplete | `DONE` |
| Origin node mismatch (for `HEAD` step) | `DONE` |
| End model unavailable (for `HEAD` step) | `DONE` |
| Job is at `HEAD` step and prefill is complete | `HEAD` |
| Job is at `HEAD` step with more prefill chunks | `EMBED` |
| Job needs layer processing | `PROCESS_LAYERS` |

---

### `EMBED`

**Purpose:** Embed the next token (or prefill chunk) to produce hidden states.

This state handles tokenization and embedding. For new jobs, it tokenizes the prompt and initializes chunking. For continuation, it embeds the most recently generated token.

**Operations:**

1. Tokenize prompt (if not already done)
2. Initialize chunking for prefill (if applicable)
3. Advance to the next chunk (if doing chunked prefill)
4. Compute embedding via `EndModel.compute_embed()`
5. Send prefill progress update (if chunking is active)

**Transitions:**

| Condition | Next State |
|-----------|------------|
| Failed to send prefill update | `DONE` |
| No model available for next layer | `DONE` |
| Next layer is virtual/remote | `SEND` |
| Next layer is local | `PROCESS_LAYERS` |

---

### `PROCESS_LAYERS`

**Purpose:** Process the job through locally-hosted model layers.

This state runs the hidden state through one or more consecutive local layer segments. Each segment processes its layer range and updates the job's `current_layer`.

**Operations:**

1. Get the local model segment for the current layer
2. Call `LlmModel.process_job()` to run through the segment's layers
3. Update the last update timestamp

**Transitions:**

| Condition | Next State |
|-----------|------------|
| No local model available | `DONE` |
| Next layer segment is remote | `SEND` |
| Next layer segment is local | `PROCESS_LAYERS` |
| All layers complete (step becomes `HEAD`) | (determined by next iteration) |

---

### `HEAD`

**Purpose:** Compute the output head to generate the next token.

This state handles the final projection and sampling step. It only runs on the **origin node** (the node that initiated the job and has the end model loaded).

**Operations:**

1. Log prefill completion (if transitioning from prefill to decode)
2. Compute RMS normalization via `EndModel.compute_norm()`
3. Compute output head projection via `EndModel.compute_head()`
4. Record timing statistics
5. If job is complete: set result and mark done
6. If more tokens needed: send update to client and continue

**Transitions:**

| Condition | Next State |
|-----------|------------|
| Job completed (EOS token generated) | `DONE` |
| Failed to send job update | `DONE` |
| More tokens to generate, next layer is local | `EMBED` |
| More tokens to generate, next layer is remote | `SEND` |
| More tokens to generate, next layer needs processing | `PROCESS_LAYERS` |

---

### `SEND`

**Purpose:** Hand off the job to another node.

This state serializes the job and sends it to the node hosting the next layer segment (or back to the origin for `HEAD` computation).

**Operations:**

1. Convert job to `NetworkJob` payload
2. Determine destination:
   - If `HEAD` step: send to origin node
   - Otherwise: send to node hosting the next layer
3. Send via `Pipe.send_job()`

**Transitions:**

| Condition | Next State |
|-----------|------------|
| Handoff complete | `DONE` |

---

### `DONE`

**Purpose:** Terminal state indicating this processing iteration is complete.

The job has either:
- Completed successfully (all tokens generated)
- Been handed off to another node
- Encountered an error condition

## State Transition Diagram

```
VALIDATING
    │
    ├──(missing job/context/pipe)────────────────────────────► DONE
    │
    ├──(HEAD step, prefill done)─────────────────────────────► HEAD
    │
    ├──(HEAD step, more prefill chunks)──────────────────────► EMBED
    │
    └──(needs layer processing)──────────────────────────────► PROCESS_LAYERS


HEAD
    │
    ├──(job complete or update failed)───────────────────────► DONE
    │
    ├──(more tokens, local embedding)────────────────────────► EMBED
    │
    ├──(more tokens, remote layer)───────────────────────────► SEND
    │
    └──(more tokens, local layer)────────────────────────────► PROCESS_LAYERS


EMBED
    │
    ├──(update failed or missing model)──────────────────────► DONE
    │
    ├──(next layer is remote)────────────────────────────────► SEND
    │
    └──(next layer is local)─────────────────────────────────► PROCESS_LAYERS


PROCESS_LAYERS
    │
    ├──(missing local model)─────────────────────────────────► DONE
    │
    ├──(next layer is remote)────────────────────────────────► SEND
    │
    └──(next layer is local)─────────────────────────────────► PROCESS_LAYERS


SEND
    │
    └──(handoff complete)────────────────────────────────────► DONE
```


## Job Context

The `JobProcessor` operates on a `JobContext` dataclass that bundles all resources needed for processing:

```python
@dataclass
class JobContext:
    config: LpConfig              # Node configuration
    logger: Optional[any]         # Logger instance
    job: Optional[Job]            # The job being processed
    pipe: Optional[Pipe]          # Pipe with local/remote segments
    end_model: Optional[EndModel] # End model (embed/norm/head)
```

## Compute Steps

The job's `compute_step` field determines what operation is needed next:

| ComputeStep | Description | Processed By |
|-------------|-------------|--------------|
| `TOKENIZE` | Convert messages to token IDs | End model (origin node) |
| `EMBED` | Embed tokens to hidden state | End model (origin node) |
| `LAYER` | Process through transformer layers | Layer segments (any node) |
| `NORM` | Apply final RMS normalization | End model (origin node) |
| `HEAD` | Project to vocabulary and sample | End model (origin node) |


## Integration Points

### Entry Point

Jobs enter the processor via `JobReceiver`, which:
1. Deserializes the `NetworkJob` payload
2. Validates the job hash
3. Creates a `JobContext`
4. Instantiates `JobProcessor` and calls `run()`

### Exit Points

Jobs exit in three ways:
1. **Completion:** Job reaches `HEAD`, generates EOS, result returned to client
2. **Handoff:** Job sent to another node via `Pipe.send_job()`
3. **Error:** Processing stops, job marked as failed


## Learn more

- [Architecture](./architecture.md) — System overview and component interactions
- [Configuration](./configuration.md) — `prefill_chunk_size` and `print_times` options
- [Privacy](./privacy.md) — How the end model provides privacy guarantees
