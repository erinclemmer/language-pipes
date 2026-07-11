---
title: Why Language Pipes?
description: An honest comparison of Language Pipes against Petals, exo, llama.cpp RPC, vLLM pipeline parallelism, and simply buying more RAM.
---

If you want to run a model that is too big for one machine, you have several good
options. This page lays out where each one wins so you can pick the right tool —
including the cases where the right tool is not Language Pipes.

## The short version

Language Pipes splits a model at transformer-layer boundaries and distributes the
segments across machines you control, over a decentralized peer-to-peer network,
behind a drop-in OpenAI-compatible API. Its distinctive properties are:

- **Current-generation architectures** (Qwen3 and Qwen3-MoE, Phi-4, Llama 2/3,
  Gemma 3/4, Ministral 3) rather than a frozen older set. See the
  [supported-models list](./model_support.md).
- **An explicit, quantified threat model** for what layer nodes can and cannot
  learn from the tensors they see see [Privacy](./privacy.md).
- **Genuinely decentralized configuration:** no central swarm coordinator, and
  each node can host its own text boundary (the End Model).
- **Python-native, drop-in OpenAI API:** point existing OpenAI client code at a
  node and keep the SDK you already use.

If those four things matter to you, Language Pipes is a good fit. If they don't,
one of the alternatives below may serve you better.

## The alternatives

### Petals

[Petals](https://github.com/bigscience-workshop/petals) pioneered the
layer-splitting, peer-to-peer approach and Language Pipes shares its core idea.
Petals is mature, has run a large public swarm, and supports collaborative
fine-tuning as well as inference — something Language Pipes does not do.

Where Language Pipes differs: Petals targets an older, fixed set of architectures
(Llama 3.1, Mixtral 8x22B, Falcon, BLOOM), and its documentation does not
quantify the privacy properties of layer-splitting or evaluate specific inversion
attacks. Language Pipes targets current models and ships an explicit threat model.
**If you need collaborative fine-tuning or want a large existing public swarm,
prefer Petals.**

### exo

[exo](https://github.com/exo-explore/exo) clusters everyday devices (laptops,
phones, Apple Silicon) into one inference pool and is excellent at automatically
partitioning a model across heterogeneous hardware with almost no configuration.
Its MLX/Apple-Silicon support is a genuine strength.

Language Pipes is less turnkey about device discovery but gives you an explicit,
file-based configuration of exactly which node hosts which layers and which node
owns the text boundary — useful when placement is a privacy decision rather than
just a performance one. **If you want zero-config clustering of mixed consumer
devices, try exo.**

### llama.cpp RPC backend

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s RPC backend shards a GGUF
model across a few machines and is the pragmatic choice for quantized `gguf`
models on CPU or mixed hardware. It is fast, lightweight, and has an enormous
ecosystem.

Language Pipes currently runs `safetensors` weights in `fp16` (with optional
int8) and does not support GGUF yet, so if you are living in the GGUF/quantized
world, llama.cpp is the better tool today. Language Pipes offers a decentralized
multi-node control plane and the OpenAI API surface out of the box. **If you want
quantized GGUF inference across a couple of boxes, use llama.cpp RPC.**

### vLLM pipeline / tensor parallelism

[vLLM](https://github.com/vllm-project/vllm) is the throughput king. Its pipeline
and tensor parallelism, paged attention, and continuous batching are built for
serving many concurrent requests fast. Typically across GPUs in one datacenter
or one well-connected cluster.

Language Pipes does **not** batch requests (see
[Failure Modes and Limitations](./architecture.md#failure-modes-and-limitations))
and is not trying to win throughput benchmarks. It targets loosely-coupled
machines you own; a couple of homes, a friend's GPU, a laptop plus a workstation —
with privacy-aware placement, not a tightly-coupled GPU cluster. **If you are
serving production traffic on GPUs you administer, use vLLM.**

### Just buy more RAM (or rent a bigger box)

Often the simplest answer. A single machine with enough memory avoids all
network complexity, all cross-node latency, and every failure mode that comes
with distribution. If one box can hold your model and renting or buying it is
within reach, do that.

Language Pipes earns its complexity only when a single sufficient box is *not*
available or *not* desirable; because the memory is spread across machines you
already have, or because you specifically want the text boundary to stay on your
own hardware while borrowing compute from someone else's.

## At a glance

| | Language Pipes | Petals | exo | llama.cpp RPC | vLLM |
|---|---|---|---|---|---|
| Multi-node split | Layer-wise, P2P | Layer-wise, P2P | Auto, heterogeneous | Layer/tensor shard | Pipeline + tensor |
| Decentralized (no coordinator) | ✓ | Public swarm | ✓ | Manual | Cluster-managed |
| Quantified threat model | ✓ | — | — | — | — |
| Current-gen architectures | ✓ | Older set | ✓ | Broad (GGUF) | Broad |
| Quantized / GGUF | int8 (no GGUF yet) | Limited | ✓ | ✓ (GGUF) | ✓ |
| Batched / high throughput | — | Limited | Limited | Limited | ✓✓ |
| Fine-tuning | — | ✓ | — | — | (training separate) |
| OpenAI-compatible API | ✓ | Partial | ✓ | ✓ (server) | ✓ |

Every project here is a reasonable choice; the honest differentiators for
Language Pipes are decentralized, privacy-aware placement of current-generation
models behind a familiar API — not raw throughput.
