---
title: Privacy
description: The privacy properties of Language Pipes, the End Model boundary, and a probabilistic threat model for hidden-state inversion attacks.
---

This document describes the privacy properties Language Pipes and the known attack surfaces.

---

## Related Work: Petals

The distributed layer-splitting architecture used by Language Pipes is similar to [Petals](https://github.com/bigscience-workshop/petals) (Borzunov et al., 2023), which enables collaborative inference and fine-tuning of large models by distributing transformer layers across a peer-to-peer network. The two projects share the same core architectural idea: split the model at layer boundaries and transmit hidden state tensors between nodes.

However, the Petals project does not provide a quantitative analysis of the privacy properties of this architecture. Its documentation acknowledges that data is "processed with the help of other people in the public swarm" and recommends private swarms for sensitive data, but does not characterize the difficulty of prompt reconstruction from intercepted hidden states, nor does it evaluate specific inversion attacks or mitigations.

Additionally, Petals targets a fixed set of older model architectures (Llama 3.1, Mixtral 8x22B, Falcon, BLOOM), several of which are no longer actively developed. Language Pipes targets current-generation open-source models; see the [supported-models list](./model_support.md) for the current, authoritative set of families and tested checkpoints.

---

## How Language Models Work (Quick Background)

To understand the End Model architecture's privacy properties, it helps to know how language models process text:

1. **Models do not process text directly.** All input must be converted to numerical representations (vectors of floating-point numbers) before the model can operate on it.

2. **Models are built in layers.** A typical model has 30-80 transformer layers stacked sequentially. Each layer takes a numerical tensor as input, applies learned linear and non-linear transformations, and passes the result to the next layer.

3. **Only the first and last components handle the text-number boundary.** The embedding layer maps tokens to vectors; the output head maps vectors back to token probabilities. All intermediate layers operate entirely in the model's latent space.

This structure is what allows Language Pipes to distribute layers across machines while maintaining a degree of privacy: **only the machine hosting the embedding and output head components directly handles text.**

## The End Model Concept

In Language Pipes, only the node hosting the **End Model** directly processes your prompts and generated responses in text form. All other nodes receive and return floating-point tensors in the model's latent representation.

### What is the End Model?

The End Model groups the components that sit at the text-number boundary:

| Component | Function |
|-----------|----------|
| **Embedding Layer** | Maps token IDs to dense vectors (text to latent space) |
| **RMS Normalization** | Normalizes the final hidden state before output projection |
| **Output Head** | Projects hidden states to vocabulary-sized logit vectors (latent space to token probabilities) |

The embedding layer maps discrete tokens into a continuous vector space, and the output head projects from that space back to a distribution over tokens. The intermediate transformer layers operate entirely within this continuous representation and never handle discrete text.

### What can layer nodes learn from hidden states?

The privacy properties of the End Model architecture depend on the difficulty of inverting hidden states back to the original prompt. Recent research has established bounds on this difficulty that warrant careful analysis.

**Theoretical invertibility.** Gupta, Basu, and Goel (2025) prove that for decoder-only Transformers with standard activation functions, the map from input token sequences to hidden state tensors is *almost-surely injective* (Theorem 2.2). That is, distinct prompts produce distinct hidden states with probability 1 under random weight initialization. This implies that hidden states are, in the information-theoretic sense, a unique encoding of the input.

**Practical recovery.** The same work provides an algorithm (SipIt) that recovers tokens sequentially from captured hidden states in O(T x |V|) worst-case time, where T is the sequence length and |V| is the vocabulary size. With gradient-guided search, practical recovery is substantially faster.

**Weight availability is assumed.** SipIt-style attacks require the model's weight matrices. Because Language Pipes is designed for open-source models hosted on platforms such as HuggingFace, the attacker should be assumed to have full access to all model weights. The threat model and mitigations described below are evaluated under this assumption.

The following sections describe the known attack vectors, their difficulty under various configurations, and the mitigations Language Pipes provides.

## Data Flow: Step by Step

### Step 1: You Enter a Prompt

You type a message to the AI. This raw text exists only on your machine.

```
"What is the capital of France?"
```

### Step 2: Tokenization

The tokenizer splits your text into sub-word units and maps each to an integer ID using a fixed vocabulary. These token IDs are a lossless, directly decodable representation of your text, so they remain local.

```
"What is the capital of France?"  ->  [ 1841, 374, 279, 6864, 315, 9822, 30 ]
```

### Step 3: Embedding

The embedding layer converts each token ID to a dense vector by table lookup in the embedding matrix. Because Language Pipes uses open-source models, the embedding matrix is available to anyone who downloads the model, making this step reversible via nearest-neighbor search against the known embedding vectors.

```
Token IDs: [ 1841, 374, 279, ... ]
                    |
Hidden states: [[ 0.023, -0.147, 0.892, ... ],    <- 4096 floats per token
                [ 0.156,  0.089, -0.445, ... ],
                ...                           ]
```

**After this step, your text and token IDs never leave your machine.** Only the hidden state tensors proceed to other nodes.

### Step 4: Data Sent to Layer Nodes

**Only these numerical arrays leave your machine:**

```
+-------------------------------------------------------------------------+
|  Hidden State Tensor: [1, 7, 4096]  <- 7 positions x 4096 floats each  |
|  Position IDs: [0, 1, 2, 3, 4, 5, 6]                                   |
|  Attention Mask: [[1, 1, 1, 1, 1, 1, 1]]                               |
|-------------------------------------------------------------------------|
|  NOT INCLUDED:                                                          |
|      - Original prompt text                                             |
|      - Token IDs                                                        |
+-------------------------------------------------------------------------+
```

Note: while the prompt text and token IDs are not transmitted, the hidden state tensor encodes sufficient information to reconstruct them given access to the model weights (see Threat Model below).

### Step 5: Layer Processing

```
+-------------------------------------------------------------------------+
|                         LAYER NODE                                      |
|-------------------------------------------------------------------------|
|  INPUT:  [1, 7, 4096] floats  ->  Transformer Layers  ->  OUTPUT: floats|
|-------------------------------------------------------------------------|
|  Without model weights: sees only floating-point arithmetic             |
|  With public model weights: can attempt prompt reconstruction           |
|     (difficulty depends on capture depth and mitigations; see below)    |
+-------------------------------------------------------------------------+
```

### Step 6: Hidden States Return (End Model Node)

The processed hidden states return from the layer nodes as floating-point tensors.

```
Returned hidden states: [[ 0.891, -0.234, 0.567, ... ],    <- Still 4096 floats per token
                         [ 0.445, 0.123, -0.789, ... ],
                         ...                         ]
```

### Step 7: Output Head (End Model Node)

The output head converts the final hidden state into a logit vector over the full vocabulary. A token is sampled from this distribution according to the request's sampling parameters.

```
Hidden state: [ 0.891, -0.234, 0.567, ... ]
                         |
              Output Head (matrix multiplication)
                         |
Token ID: 12366  (chosen based on scores and request parameters)
```

### Step 8: Decoding (End Model Node)

The tokenizer maps the selected token ID back to text. This output text exists only on your machine.

```
Token ID: 12366  ->  "Paris"
```

**The generated response never leaves your machine.** Only the End Model node sees the final text output.

---

## Threat Model

The End Model architecture provides layered privacy whose strength depends on the deployment configuration and the capabilities of the adversary. This section characterizes the known attack vectors and quantifies the difficulty of each under various mitigation configurations.

### Attack Vectors

#### 1. Embedding Inversion (Layer 0 Capture)

If hidden states are transmitted immediately after embedding (capture at layer 0), the attacker performs a nearest-neighbor lookup against the public embedding matrix to recover token IDs. This is a trivial O(T x |V|) operation with near-perfect accuracy.

**Baseline recovery probability: ~100%** at layer 0.

**Mitigation:** Retaining the first N transformer layers on the End Model node by setting the `LP_NUM_LOCAL_LAYERS` environment variable eliminates layer-0 exposure entirely. `LP_NUM_LOCAL_LAYERS` defaults to `1`, which is enough to remove the trivial layer-0 attack; for privacy-sensitive deployments we recommend raising it so more of the early pipeline stays on your machine. All nodes hosting the same end model must use the same value.

#### 2. SipIt Deep-Layer Recovery

The SipIt algorithm (Gupta, Basu, and Goel, 2025) recovers tokens sequentially by replaying candidate embeddings through layers 0 to L and comparing against the captured hidden state. Its worst-case running time is `O(T × |V|)` — polynomial in the sequence length T and vocabulary size |V| so deeper capture does not change the asymptotic hardness of recovery. What it does change is the *practical* cost: as hidden states pass through more layers, `float16` rounding accumulates and the mapping back to token-space becomes numerically harder to invert exactly, so a successful attack needs more compute and more careful search per token.

We expect recovery to become harder with capture depth for these reasons, **but we have not measured this.** Quantifying how recovery success rate falls off with depth (e.g. capturing hidden states at layers 1, 5, 10, and 20 and reporting recovery rates) is the experiment this threat model does not yet contain; treat the "harder with depth" expectation as unverified until it is run. Do not rely on depth alone as a security boundary.

**Mitigation:** Network security is the best way to avoid attacks. Assume that a determined attacker with the model weights can invert the hidden states passed between nodes, and keep layer nodes on machines you trust.

#### 3. Network Eavesdropping

Without AES encryption (`network_key`), hidden state tensors travel as unencrypted HTTP payloads. Any observer on the network path can capture them.

**Mitigation:** Configure `network_key` to enable AES encryption of all network traffic. See the [Configuration Manual](./configuration.md).

#### 4. Unauthenticated API Access

The OpenAI-compatible API server binds to `0.0.0.0` with no authentication by default. An attacker on the local network can submit prompts, which may be useful for known-plaintext attacks against the AES encryption layer.

**Mitigation:** Restrict API access via firewall rules or bind to localhost when the API is not intended for external use.

#### 5. Malicious Layer Node Returning Bad Tensors (Integrity)

The attack vectors above concern **confidentiality**. Can a layer node read your prompt? They do not address **integrity**. Can a layer node lie to you?

A hostile layer node is free to return whatever tensor it likes. It can compute garbage, or, more insidiously, a *subtly steered* hidden state that biases the final output in a chosen direction, and then hash the result correctly. The SHA-256 check on each `NetworkJob` validates **transport integrity** (the bytes were not corrupted or tampered with in transit); it does **not** validate **computational honesty** (that the node actually ran the correct layers on the correct weights). A node that deliberately returns a manipulated tensor produces a valid hash for that manipulated tensor.

In a genuinely open peer-to-peer swarm this is arguably a larger problem than prompt recovery: a malicious node can influence generated outputs without ever being detected, and there is currently no cross-checking, redundant recomputation, or attestation that would catch it.

**Mitigation:** Known gap. Currently mitigated only by trusting your layer operators. Run layer nodes on machines you control or trust, or restrict the swarm to whitelisted node IDs (`whitelist_node_ids`). Language Pipes does not yet provide redundant recomputation or verifiable-computation guarantees against a dishonest layer node.

### Mitigation Stack Summary

The following table summarizes the available mitigations, their effectiveness against each attack vector, and their configuration:

| Mitigation | SipIt (confidentiality) | Network Eavesdrop | Bad tensors (integrity) | Configuration |
|------------|-------|--------------------|-------|---------------|
| **Architectural separation** (always on) | Prevents casual observation; does not prevent deliberate inversion | Same | No effect — a node still controls its own output | Default behavior |
| **Local layers** (`LP_NUM_LOCAL_LAYERS`) | Removes layer-0 exposure and keeps early layers on your machine; raises practical inversion cost | No effect | No effect | `LP_NUM_LOCAL_LAYERS` env var (default `1`) |
| **AES encryption** (`network_key`) | Does not apply (malicious node decrypts to compute) | **Effective.** Encrypts transport | No effect | `network_key` in config |
| **Trusted layer nodes** | **Effective.** Trusted operator should not attempt inversion | **Effective.** Same | **Effective** — the only current defense; trusted operator should not return manipulated tensors | Deploy layer nodes only on trusted machines; `whitelist_node_ids` |

### Probabilistic Security Summary

The `LP_NUM_LOCAL_LAYERS` environment variable defaults to `1`. For privacy-sensitive
deployments we recommend **raising it** so more of the early pipeline stays on your
machine; the summary below assumes AES encryption plus a raised `LP_NUM_LOCAL_LAYERS`
(all nodes hosting the same end model must use the same value):

| Attack Vector | Estimated Difficulty | Notes |
|---------------|---------------------|-------|
| Casual observation by layer operator | **Infeasible** | Layer nodes see only floating-point tensors |
| Network eavesdropping | **Infeasible** | AES-encrypted transport |
| SipIt prompt recovery | **Possible** | The algorithm is polynomial — `O(T × \|V\|)` — in sequence length and vocabulary size. The practical barrier is float precision and search cost, not asymptotic hardness. Given enough compute and time, a layer operator with the model weights can succeed. |
| Output tampering by layer node | **Possible** | Undetectable without trusting the operator; the SHA-256 check does not verify computational honesty |

With recommended mitigations **plus trusted layer node operators** — i.e. you also
control or trust every machine hosting layer segments — the residual risk changes
substantially:

| Attack Vector | Estimated Difficulty | Notes |
|---------------|---------------------|-------|
| Casual observation by layer operator | **Infeasible** | As above |
| Network eavesdropping | **Infeasible** | As above |
| SipIt prompt recovery | **Infeasible in practice** | A trusted operator with the capability to invert has no incentive to; the attack requires an operator who is both capable and hostile |
| Output tampering by layer node | **Infeasible in practice** | Same — tampering requires a hostile operator |

Trusting the layer operators collapses the confidentiality *and* integrity risks to
the operators themselves, which is why hosting layer nodes only on machines you trust
is the strongest single mitigation available today.

---

### Privacy Enhancement: Host the End Model Yourself

```
+-----------------+
|   Your Machine  | <-- End Model: prompts stay here
|   Layers 0-10   |
+--------+--------+
         | Hidden states only (AES encrypted)
         v
+-----------------+
|  Friend's GPU   | <-- Layers only: does not see prompts
|   Layers 11-31  |
+-----------------+
```

```toml
# Your machine
end_models = ["Qwen/Qwen3-1.7B"]  # You control the End Model

[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cpu"
memory = 4

# Friend's machine
[[layer_models]]
model_id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
memory = 8
```

Your friend can choose to also host an end model, but only your machine will see your prompts if you send requests to your machine. Enable `network_key` for AES-encrypted transport between nodes.

## References

- Gupta, Basu, and Goel. "Transformers are Injective: SipIt Sequential Prompt Inversion from Intermediate Representations." 2025. [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)
- Borzunov et al. "Petals: Collaborative Inference and Fine-tuning of Large Models." ACL 2023. [arXiv:2209.01188](https://arxiv.org/abs/2209.01188)
