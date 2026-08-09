---
title: Job Processor State Machine
description: The state machine that controls the execution of jobs in the distributed inference pipeline. It gives the states, the transitions, and the integration points.
---

The `JobProcessor` class is a finite state machine (FSM). The FSM controls the execution of jobs in the distributed inference pipeline. This document gives each state, the conditions for each transition, and the integration points with Language Pipes.


## Overview

A job comes to a node through the `JobReceiver`. The `JobReceiver` gives the job to a `JobProcessor` instance. The processor makes sure that the job context is correct. Then the processor sends the computation to local or remote model segments. At the end, the processor completes the job or sends the job to a different node.


## States

### `VALIDATING`

**Purpose:** Make sure that the job context is correct before the processor starts the job.

The FSM starts in this state for each job. The state makes sure that the job object exists. For a `HEAD` compute step, the state also makes sure that:

- The origin node is the local node. Only the origin node computes the head.
- The end model is loaded.

Then the state finds the next state. The compute step of the job and the location of the current layer give the next state.

**Transitions:**

| Condition | Next State |
|-----------|------------|
| The job is missing | `DONE` |
| The step is `HEAD`, and the origin node is not the local node | `DONE` |
| The step is `HEAD`, and the end model is not available | `DONE` |
| No node in the pipe has the current layer | `DONE` |
| The step is `EMBED` or `TOKENIZE`, and the origin node is not the local node | `SEND` |
| The node for the current layer is remote | `SEND` |
| The step is `HEAD`, and the prefill is complete | `HEAD` |
| The step is `EMBED` or `TOKENIZE`, and the origin node is the local node | `EMBED` |
| The step is `HEAD`, and more prefill chunks remain | `EMBED` |
| The current layer is 0, and the end model has local layers | `PROCESS_LAYERS` |
| The node for the current layer is local | `PROCESS_LAYERS` |

---

### `EMBED`

**Purpose:** Embed the next token or the next prefill chunk to compute the hidden state.

This state tokenizes and embeds. For a new job, the state tokenizes the prompt and initializes the chunking. For a job that continues, the state embeds the last token that the head computed.

**Operations:**

1. The state tokenizes the prompt, if the prompt is not tokenized.
2. The state initializes the chunking for the prefill, if the chunking is applicable.
3. The state moves to the next chunk, if the prefill is chunked.
4. The state computes the embedding with `EndModel.compute_embed()`.
5. The state sends a prefill progress update, if the chunking is active.

**Transitions:**

| Condition | Next State |
|-----------|------------|
| The end model is not available | `DONE` |
| The state cannot send the prefill update | `DONE` |
| No node in the pipe has the next layer | `DONE` |
| The next layer is remote | `SEND` |
| The next layer is local | `PROCESS_LAYERS` |

---

### `PROCESS_LAYERS`

**Purpose:** Process the job through the model layers on the local node.

This state sends the hidden state through one or more local layer segments. The segments are consecutive. Each segment computes its range of layers. Then the segment updates the `current_layer` field of the job.

**Operations:**

1. The state gets the local model segment for the current layer.
2. The state calls `LlmModel.process_job()` to compute the layers of the segment.
3. The state updates the timestamp of the last update.

**Transitions:**

| Condition | Next State |
|-----------|------------|
| No node in the pipe has the current layer | `DONE` |
| The next layer segment is remote | `SEND` |
| All layers are complete, and the origin node is not the local node | `SEND` |
| The next layer segment is local | `PROCESS_LAYERS` |
| All layers are complete, the origin node is the local node, and the prefill is complete | `HEAD` |
| All layers are complete, the origin node is the local node, and more prefill chunks remain | `EMBED` |

---

### `HEAD`

**Purpose:** Compute the output head to generate the next token.

This state computes the final projection and samples the next token. The state operates only on the **origin node**. The origin node started the job and has the end model.

**Operations:**

1. The state writes a log entry when the prefill is complete and the decode starts.
2. The state computes the RMS normalization with `EndModel.compute_norm()`.
3. The state computes the output head projection with `EndModel.compute_head()`.
4. The state records the timing statistics.
5. If the job is complete, the state sets the result and marks the job as done.
6. If more tokens are necessary, the state sends an update to the client. Then the processor continues.

**Transitions:**

| Condition | Next State |
|-----------|------------|
| The end model is not available | `DONE` |
| More prefill chunks remain | `DONE` |
| The job is complete | `DONE` |
| The state cannot send the job update | `DONE` |
| More tokens are necessary | `EMBED` |

**NOTE:** This state always operates on the origin node, and the origin node has the end model. Thus the next state is `EMBED`, and never `SEND` or `PROCESS_LAYERS`.

A job is complete when the model generates the EOS token, or when the token count comes to `max_completion_tokens`.

---

### `SEND`

**Purpose:** Send the job to a different node.

This state converts the job to a network payload. Then the state sends the payload to the node that has the next layer segment. For a `HEAD` step, the state sends the payload to the origin node.

**Operations:**

1. The state converts the job to a `NetworkJob` payload.
2. The state finds the destination:
   - For a `HEAD` step, the destination is the origin node.
   - For all other steps, the destination is the node that has the next layer.
3. The state sends the payload with `Pipe.send_job()`.

**Transitions:**

| Condition | Next State |
|-----------|------------|
| The state sent the job | `DONE` |
| No node in the pipe has the next layer | `DONE` |

---

### `DONE`

**Purpose:** The terminal state. This state shows that the current iteration is complete.

The job is in one of these three conditions:

- The job is complete, and the model generated all tokens.
- A different node received the job.
- An error condition stopped the job.

---

## Cancellation

A pipe can lose a piece while a job runs on it. An operator unloads a model from
the TUI, or a node that hosts a segment leaves the network. The job cannot
finish after that.

Every transition to `DONE` that comes from a missing piece cancels the job:

- No node in the pipe has the current or the next layer.
- The end model of the origin node is not available.

Cancellation does these operations:

1. The processor marks the job with a reason. `run()` checks the reason before
   each state, so a job that a different thread cancels stops at the next state
   boundary.
2. The `JobTracker` removes the job from the pending jobs and resolves the
   promise of the caller. An API client gets an error instead of an open
   request.
3. The node sends a `JobCancel` packet to the origin node of the job, if the
   origin node is a different node. The origin node holds the API request, so
   the origin node must know. A node that gets a `JobCancel` for a job that
   started on a different node sends the packet on toward the origin node.

The `ModelManager` cancels jobs before it frees the tensors of a model:

| Operation | Canceled jobs |
|-----------|---------------|
| `shutdown_layer_models` | Every job on a pipe that the unloaded segments belong to |
| `shutdown_end_model` | Every job that this node started for that model |

Jobs that a different node started do not use the end model of this node, so
`shutdown_end_model` does not cancel them.

Without cancellation, a job stays in the pending jobs until `EXPIRED_JOB_TIME`
(60 seconds) passes, and the API client waits for the whole time.

## State Transition Diagram

```
VALIDATING
    │
    ├──(no job, or no node for the current layer)────────────► DONE
    │
    ├──(`HEAD` step, not the origin node)────────────────────► DONE
    │
    ├──(`HEAD` step, no end model)───────────────────────────► DONE
    │
    ├──(`EMBED`/`TOKENIZE` step, not the origin node)────────► SEND
    │
    ├──(the node for the current layer is remote)────────────► SEND
    │
    ├──(`HEAD` step, prefill complete)───────────────────────► HEAD
    │
    ├──(`EMBED`/`TOKENIZE` step, or more prefill chunks)─────► EMBED
    │
    └──(the node for the current layer is local)─────────────► PROCESS_LAYERS


HEAD
    │
    ├──(no end model, or more prefill chunks)────────────────► DONE
    │
    ├──(job complete, or update failed)──────────────────────► DONE
    │
    └──(more tokens)─────────────────────────────────────────► EMBED


EMBED
    │
    ├──(no end model, or update failed)──────────────────────► DONE
    │
    ├──(no node for the next layer)──────────────────────────► DONE
    │
    ├──(next layer is remote)────────────────────────────────► SEND
    │
    └──(next layer is local)─────────────────────────────────► PROCESS_LAYERS


PROCESS_LAYERS
    │
    ├──(no node for the current layer)───────────────────────► DONE
    │
    ├──(next layer is remote)────────────────────────────────► SEND
    │
    ├──(all layers complete, not the origin node)────────────► SEND
    │
    ├──(next layer is local)─────────────────────────────────► PROCESS_LAYERS
    │
    ├──(all layers complete, prefill complete)───────────────► HEAD
    │
    └──(all layers complete, more prefill chunks)────────────► EMBED


SEND
    │
    └──(job sent, or no node for the next layer)─────────────► DONE
```

## Compute Steps

The `compute_step` field of the job gives the next necessary operation:

| ComputeStep | Description | Computed by |
|-------------|-------------|-------------|
| `TOKENIZE` | Converts the messages to token IDs | The end model (origin node) |
| `EMBED` | Embeds the tokens to give the hidden state | The end model (origin node) |
| `LAYER` | Computes the transformer layers | The layer segments (any node) |
| `NORM` | Applies the final RMS normalization | The end model (origin node) |
| `HEAD` | Projects to the vocabulary and samples the next token | The end model (origin node) |


## Integration Points

### Entry Point

Jobs come to the processor through the `JobReceiver`. The `JobReceiver` does these operations:

1. It deserializes the `NetworkJob` payload.
2. It makes sure that the job hash is correct.
3. It creates a `JobContext`.
4. It creates a `JobProcessor` instance and calls `run()`.

### Exit Points

A job exits the processor in one of three ways:

1. **Completion:** The job comes to the `HEAD` state. The model generates the EOS token. The processor sends the result to the client.
2. **Send:** The processor sends the job to a different node with `Pipe.send_job()`.
3. **Error:** The processor stops, and the processor marks the job as failed. If
   the pipe lost a piece that the job needs, the processor also cancels the job.
   See [Cancellation](#cancellation).
