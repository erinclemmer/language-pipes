# Privacy Architecture

Language Pipes provides privacy-preserving distributed inference through its **End Model** architecture. This document describes the privacy properties of the system, the known attack surfaces, and the probabilistic guarantees each mitigation provides.

---

## Related Work: Petals

The distributed layer-splitting architecture used by Language Pipes is similar to [Petals](https://github.com/bigscience-workshop/petals) (Borzunov et al., 2023), which enables collaborative inference and fine-tuning of large models by distributing transformer layers across a peer-to-peer network. The two projects share the same core architectural idea: split the model at layer boundaries and transmit hidden state tensors between nodes.

However, the Petals project does not provide a quantitative analysis of the privacy properties of this architecture. Its documentation acknowledges that data is "processed with the help of other people in the public swarm" and recommends private swarms for sensitive data, but does not characterize the difficulty of prompt reconstruction from intercepted hidden states, nor does it evaluate specific inversion attacks or mitigations.

A primary objective of Language Pipes is to fill this gap: to rigorously measure and document the probabilistic privacy guarantees that a layer-splitting architecture can and cannot provide. The threat model, empirical recovery measurements, and mitigation analysis presented in this document and the accompanying case study [SipIt](./threat-model/sipit.md) represent this effort.

Additionally, Petals targets a fixed set of older model architectures (Llama 3.1, Mixtral 8x22B, Falcon, BLOOM), several of which are no longer actively developed. Language Pipes targets current-generation open-source models, beginning with the Qwen3 and Qwen3-MoE architectures.

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

**Mitigation:** Retaining the first N transformer layers on the End Model node (first-N local layers) eliminates layer-0 exposure entirely.

#### 2. SipIt Deep-Layer Recovery

The SipIt algorithm (Gupta, Basu, and Goel, 2025) recovers tokens sequentially by replaying candidate embeddings through layers 0 to L and comparing against the captured hidden state. Recovery difficulty increases exponentially with capture depth due to float16 precision accumulation, non-convex optimization landscapes, and computational cost scaling.

Empirical measurements show recovery dropping from ~100% at L0 to ~4% at L5, with extrapolated recovery below 1% by approximately L8.

**Mitigation:** First-N local layers (N >= 5) reduces SipIt recovery to near-zero.

For the full analysis including per-prompt recovery data, timing measurements, and the interaction between capture depth and float16 precision, see the [SipIt case study](./threat-model/sipit.md).

#### 3. Network Eavesdropping

Without AES encryption (`network_key`), hidden state tensors travel as unencrypted HTTP payloads. Any observer on the network path can capture them.

**Mitigation:** Configure `network_key` to enable AES encryption of all network traffic. See the [Configuration Manual](./configuration.md).

#### 4. Unauthenticated API Access

The OpenAI-compatible API server binds to `0.0.0.0` with no authentication by default. An attacker on the local network can submit prompts, which may be useful for known-plaintext attacks against the AES encryption layer.

**Mitigation:** Restrict API access via firewall rules or bind to localhost when the API is not intended for external use.

### Mitigation Stack Summary

The following table summarizes the available mitigations, their effectiveness against each attack vector, and their configuration:

| Mitigation | SipIt | Network Eavesdrop | Configuration |
|------------|-------|--------------------|---------------|
| **Architectural separation** (always on) | Prevents casual observation; does not prevent deliberate inversion | Same | Default behavior |
| **AES encryption** (`network_key`) | Does not apply (malicious node decrypts to compute) | Same | `network_key` in config |
| **First-N local layers** | **Effective.** Recovery drops exponentially with N | No effect (attacker completes forward pass) | Planned: `local_layer_count` |
| **Trusted layer nodes** | **Effective.** Trusted operator will not attempt inversion | **Effective.** Same | Deploy layer nodes only on trusted machines |

### Probabilistic Security Summary

With recommended mitigations (AES encryption + first-5 local layers):

| Attack Vector | Estimated Difficulty | Notes |
|---------------|---------------------|-------|
| Casual observation by layer operator | **Infeasible** | Layer nodes see only floating-point tensors |
| Network eavesdropping | **Infeasible** | AES-encrypted transport |
| SipIt prompt recovery | **Impractical** | ~4% token recovery at L5 baseline |

With recommended mitigations + **trusted layer node operators**:

| Attack Vector | Estimated Difficulty |
|---------------|---------------------|
| All inversion attacks | **Infeasible** trusted operator should not attempt reconstruction |

---

## Deployment Patterns

### Standard Privacy: Host the End Model Yourself

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
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4

end_models = ["Qwen/Qwen3-1.7B"]  # You control the End Model

# Friend's machine
[[layer_models]]
id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
max_memory = 8
```

Your friend can choose to also host an end model, but only your machine will see your prompts if you send requests to your machine. Enable `network_key` for AES-encrypted transport between nodes.

### Enhanced Privacy: First-N Local Layers + Encryption

For sensitive workloads with public models, retain the first N transformer layers locally. This eliminates trivial embedding inversion and reduces SipIt recovery to low single-digit percentages:

```
+-----------------------+
|     Your Machine      | <-- End Model + first 5 layers
|   Embed, L0-L4, Norm  |
|   Head                |
+-----------+-----------+
            | Hidden states at L5 (AES encrypted)
            v
+-----------------------+
|     Remote GPU(s)     | <-- Layers 5-31 only
+-----------------------+
```

### Maximum Privacy: Trusted Network Only

For the strongest protection, deploy layer nodes exclusively on machines operated by trusted parties. Combined with AES encryption and first-N local layers, this configuration addresses all known attack vectors: the architectural and computational mitigations raise the bar against passive and semi-active adversaries, while the trust relationship eliminates the motivated adversary scenario entirely.

---

## FAQ

**Can layer nodes reconstruct my prompts?**
A layer node operator with access to the public model weights can, in principle, apply inversion algorithms to recover prompt tokens from captured hidden states. The difficulty of this reconstruction depends on the capture depth and active mitigations: with first-N local layers (N >= 5), SipIt-style recovery drops to low single-digit percentages. For the strongest guarantee, deploy layer nodes only on trusted machines.

**What if someone captures the hidden states?**
Hidden states encode sufficient information to reconstruct the input prompt when combined with the model weights (Gupta et al., 2025). Treat hidden states as sensitive data. Always enable `network_key` encryption for network transport, use first-N local layers to increase inversion difficulty, and deploy layer nodes only on trusted machines.

**Is this encryption?**
No. The End Model architecture provides *architectural separation*. Layer nodes operate on continuous-valued tensors rather than discrete text. This is not cryptographic protection: the tensors are theoretically invertible given the model weights. For cryptographic transport security, configure `network_key` to enable AES encryption. The combination of architectural separation and AES encryption provides defense in depth.

**How does this compare to running the full model locally?**
Running the full model on a single machine provides complete privacy and no data leaves the machine. Distributed inference trades some privacy for computational scalability. The mitigations described above narrow this gap, but do not fully close it when layer nodes are operated by untrusted parties.

---

## References

- Gupta, Basu, and Goel. "Transformers are Injective: SipIt Sequential Prompt Inversion from Intermediate Representations." 2025. [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)
- Borzunov et al. "Petals: Collaborative Inference and Fine-tuning of Large Models." ACL 2023. [arXiv:2209.01188](https://arxiv.org/abs/2209.01188)

### Case Studies
* [SipIt: Sequential Prompt Inversion from Hidden States](./threat-model/sipit.md)
* More to come

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [Open AI Compatible API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [The default peer to peer implementation](./distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./llm-layer-collector.md)
