# Case Study: SipIt Sequential Prompt Inversion from Hidden States

This document provides a detailed analysis of the SipIt attack against Language Pipes' distributed inference architecture. For the general threat model overview, see [Privacy Architecture](../privacy.md).

---

## Overview

SipIt (Gupta, Basu, and Goel, 2025) demonstrates that the map from input token sequences to Transformer hidden states is *almost-surely injective* i.e. distinct prompts produce distinct hidden states with probability 1. The paper provides both a theoretical proof (Theorem 2.2) and a practical recovery algorithm that reconstructs prompt tokens from captured hidden states.

This is directly relevant to Language Pipes because hidden state tensors are the primary data transmitted between nodes during distributed inference. A malicious layer node operator who captures these tensors can apply SipIt to recover the original prompt.

## How SipIt Works

### Theoretical Foundation

For a decoder-only Transformer with standard (analytic) activation functions, the map:

```
f: token sequence s -> hidden state matrix H^(l)(s) at layer l
```

is almost-surely injective. That is, for almost all weight initializations, if `f(s1) = f(s2)` then `s1 = s2`. This holds at *any* layer l, meaning a single layer's hidden states are theoretically sufficient to uniquely identify the input prompt.

### Recovery Algorithm

SipIt exploits the causal (autoregressive) structure of decoder-only Transformers. Because position t's hidden state depends only on tokens at positions 0 through t, the algorithm recovers tokens sequentially:

```
Input:  captured hidden states H at layer L
        public model weights for layers 0..L
        vocabulary V

For t = 0 to sequence_length - 1:
    For each candidate token v in V:
        1. Construct prefix: [recovered_0, ..., recovered_{t-1}, v]
        2. Embed the prefix using the public embedding matrix
        3. Run the embedded prefix through layers 0..L
        4. Compare: if ||predicted_H[t] - captured_H[t]|| < epsilon
              then recovered_t = v; break
```

In the worst case, this requires O(T x |V|) forward passes, where T is the sequence length and |V| is the vocabulary size. In practice, gradient-guided search reduces the per-position cost substantially by optimizing a continuous relaxation of the token embedding and snapping to the nearest discrete embedding.

### What the Attacker Needs

From Language Pipes' wire format (`JobData` and `NetworkJob`), a malicious layer node captures:

| Field | Content | Role in Attack |
|-------|---------|----------------|
| `JobData.state` | Hidden state tensor `[1, seq_len, hidden_dim]` (float16) | **Primary target** the captured H matrix |
| `current_layer` | Layer boundary index | Tells the attacker L (the capture depth) |
| `position_ids` | Sequence position information | Required for positional encoding during replay |
| `position_embeddings` | Rotary cos/sin values | Allows exact replay without recomputation |
| `pipe_id` | Model identifier | Identifies which public model weights to download |

All model weights are available from the same public HuggingFace repository that Language Pipes itself downloads from.

## Recovery Probability by Capture Depth

The theoretical injectivity proof assumes real-valued (infinite precision) arithmetic. In practice, Language Pipes transmits hidden states in float16, which introduces quantization effects that degrade recovery at deeper capture points.

### Why Deeper Capture Degrades SipIt

Three factors compound with increasing capture depth L:

1. **Float16 precision accumulation.** Each transformer layer applies attention, normalization, and feed-forward operations that accumulate rounding errors. The minimum pairwise L2 distance between hidden states for distinct tokens shrinks with depth, approaching the float16 noise floor (~4.88e-4 machine epsilon). At shallow layers, distances are large (direct embedding lookups). At deeper layers, quantization causes meaningful collisions between candidate representations.

2. **Non-convex optimization landscape.** The gradient-guided search variant of SipIt optimizes a continuous loss:

   ```
   minimize ||forward(prefix + embed_t, layers 0..L) - captured[t]||^2
   ```

   With more layers in the forward pass, this loss landscape becomes increasingly non-convex. Attention operations create sharp, data-dependent routing with many local minima. The gradient shortcut is most reliable at shallow depths where the embedding-to-hidden-state mapping is nearly linear.

3. **Computational cost.** Each candidate token evaluation requires a full forward pass through all L layers. Cost scales linearly with L:

   | Capture Layer | Mean Time Per Prompt (PoC hardware) |
   |---------------|-------------------------------------|
   | L0            | < 0.1s                              |
   | L1            | ~36s                                |
   | L2            | ~67s                                |
   | L3            | ~98s                                |
   | L4            | ~130s                               |
   | L5            | ~160s                               |

### Empirical Recovery Rates

Measurements on Qwen3-0.6B (28 layers, hidden_dim=1024) with no mitigations other than capture depth:

| Capture Layer | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 | Mean Recovery |
|---------------|----------|----------|----------|----------|----------|---------------|
| L0            | 100%     | 100%     | 100%     | 100%     | 100%     | **100.0%**    |
| L1            | 75%      | 54%      | 32%      | 42%      | 44%      | **49.4%**     |
| L2            | 33%      | 25%      | 36%      | 26%      | 39%      | **31.8%**     |
| L3            | 25%      | 8%       | 5%       | 26%      | 17%      | **16.2%**     |
| L4            | 25%      | 8%       | 9%       | 13%      | 11%      | **13.2%**     |
| L5            | 8%       | 4%       | 0%       | 10%      | 0%       | **4.4%**      |

Recovery drops approximately exponentially with depth:

```
L0: 100.0%  ||||||||||||||||||||||||||||||||||||||||||||||||||||
L1:  49.4%  |||||||||||||||||||||||||
L2:  31.8%  ||||||||||||||||
L3:  16.2%  ||||||||
L4:  13.2%  |||||||
L5:   4.4%  ||
```

Extrapolation: recovery drops below 1% by approximately L8, and approaches 0% by L10+. For larger models (70B+ with 80+ layers), the decay is expected to be steeper due to higher-dimensional hidden spaces and more complex attention routing.

## Mitigation: First-N Local Layers

Retaining the first N transformer layers on the End Model node increases the capture depth from L0 to LN. This is the primary defense against SipIt.

**Mechanism:** The End Model node runs layers 0 through N-1 locally before transmitting the hidden state. Remote nodes receive `JobData.state` at `current_layer = N` instead of 0.

**Effectiveness:** At N=5, SipIt recovery drops from ~100% to ~4%. The performance cost depends on the model:


For large models, N=5-10 is a modest overhead. For small models, it consumes a more significant fraction of available compute.

**Configuration:** Set `num_local_layers` in the configuration file or interactive settings (default N=1).

**Quality impact:** No difference in inference compution.

**Resource impact:** Each end node must load N layers into memory. This increases the minimum RAM requirements.

## References

- Gupta, Basu, and Goel. "Transformers are Injective: SipIt Sequential Prompt Inversion from Intermediate Representations." 2025. [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)

### Documentation
* [CLI Reference](../cli.md)
* [Privacy Protection](../privacy.md)
* [Configuration Manual](../configuration.md)
* [Architecture Overview](../architecture.md)
* [Open AI Compatible API](../oai.md)
* [Job Processor State Machine](../job-processor.md)
* [The default peer to peer implementation](../distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](../llm-layer-collector.md)
