---
title: Why Language Pipes?
description: An honest comparison of Language Pipes against Petals, exo, llama.cpp RPC, vLLM pipeline parallelism, and simply buying more RAM.
---

If you want to run a model that is too big for one machine, you have several good
options. This page lays out where each one wins so you can pick the right tool.

## Language Pipes Features

Language Pipes splits a model at transformer-layer boundaries and distributes the
segments across machines you control, over a decentralized peer-to-peer network,
behind a drop-in OpenAI-compatible API. Its distinctive properties are:

**Support for Current Generation Models:** This project is being actively updated to support the latest models. Since this is a one person project not all models are supported but more are added with almost every release. See the [supported-models list](./model_support.md) for more information on what is currently supported.

**Quantified Threat Model:** Security for anonymous distributed LLM inference is still an open problem in science with every solution having some severe trade off. This project is dedicated to finding the best practical solution and backing that up with research. See the [Privacy](./privacy.md) documentation for more information.

**Based on Huggingface Transformers:** The transformers library allows Language Pipes to quickly add support for models without writing the entire model stack from scratch. This means easy support for Hugginface supported models when they are added, but it does not work on GGUF quantized formats. 

**Decentralization Specialization:** Language Pipes specializes in being a completely decentralized application designed to take advantage of the latest proven technology stack in decentralized LLM inference.

**Python Native:** This project is written entirely in Python, from the TUI to the actual LLM generation code. This means if you can get Pytorch installed on your machine you are most of the way to getting this working. No need to install Node.js or any other language on your machine.

**OpenAI Compabatable API:** Supports standard `/v1/chat/completions` and `/v1/responses` endpoints for easy drop in to existing technology stacks.

If those matter to you, Language Pipes is a good fit. If they don't,
one of the alternatives below may serve you better.

## The alternatives

### llama.cpp RPC backend

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s RPC backend shards a GGUF
model across a few machines and is the pragmatic choice for quantized `gguf`
models on CPU or mixed hardware. It is fast, lightweight, and has an enormous
ecosystem.

Language Pipes currently runs `safetensors` weights in `fp16` (with optional
int8) and does not support GGUF yet, so if you are living in the GGUF/quantized
world, llama.cpp is the better tool today. Language Pipes offers a decentralized
multi-node control plane and the OpenAI API surface out of the box.  
**If you want quantized GGUF inference across a couple of boxes with centralized control, use llama.cpp RPC.**

### vLLM pipeline / tensor parallelism

[vLLM](https://github.com/vllm-project/vllm) is the throughput king. Its pipeline
and tensor parallelism, paged attention, and continuous batching are built for
serving many concurrent requests fast. Typically across GPUs in one datacenter
or one well-connected cluster.

Language Pipes does **not** batch requests (see
[Failure Modes and Limitations](./architecture.md#failure-modes-and-limitations))
and is not trying to win throughput benchmarks. It targets loosely-coupled
machines you own; a couple of homes, a friend's GPU, a laptop plus a workstations
with privacy-aware placement, not a tightly-coupled GPU cluster.  
**If you are serving production traffic on GPUs you administer, use vLLM.**

### Exo

[exo](https://github.com/exo-explore/exo) clusters everyday devices (laptops,
phones, Apple Silicon) into one inference pool and is excellent at automatically
partitioning a model across heterogeneous hardware with almost no configuration. It is not truly decentralized though and requires either exposing token IDs with tensor-parallelism or the output token in pipeline-parallelism. Its MLX/Apple-Silicon support is a genuine strength, but also support for non-Apple platforms is limited for linux and absent completely for Windows.

Language Pipes is less turnkey about device discovery but gives you an explicit,
file-based configuration of exactly which node hosts which layers and which node
owns the text boundary which is useful when placement is a privacy decision rather than
just a performance one.  
**If you want zero-config clustering of mixed consumer devices that all happen to run MacOS and no inter-network privacy expectations, try exo.**

### Petals

[Petals](https://github.com/bigscience-workshop/petals) pioneered the
layer-splitting, peer-to-peer approach and Language Pipes shares its core idea.
Petals is mature and supports collaborative fine-tuning as well as inference, but does has not seen updates in years.

Where Language Pipes differs: Petals targets an older, fixed set of architectures
(Llama 3.1, Mixtral 8x22B, Falcon, BLOOM), and its documentation does not
quantify the privacy properties of layer-splitting or evaluate specific inversion
attacks. Language Pipes targets current models and ships an explicit threat model.  
**Petals is mentioned here for historical purposes and is not widely used.**

### Just buy more RAM

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
| Decentralized (no coordinator) | ✓ | Public swarm | Master Managed | Manual | Cluster-managed |
| Current-gen architectures | ✓ | Older set | ✓ | Broad (GGUF) | Broad |
| Quantized / GGUF | int8 (no GGUF) | Limited | ✓ | ✓ (GGUF) | ✓ |
| Batched / high throughput | — | Limited | Limited | Limited | ✓✓ |
| Fine-tuning | — | ✓ | — | — | (training separate) |
| OpenAI-compatible API | ✓ | Partial | ✓ | ✓ (server) | ✓ |