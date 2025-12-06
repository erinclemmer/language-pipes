# Privacy Architecture

Language Pipes provides privacy-preserving distributed inference through its **End Model** architecture. This document explains how your prompt data stays protected when using a distributed network.

---

## The End Model Concept

When you split a language model across multiple machines, a natural question arises: **who sees your data?**

In Language Pipes, only the node hosting the **End Model** ever sees your actual prompts and responses. All other nodes in the network process numerical tensors—high-dimensional vectors that are unintelligible without the full model context.

### What is the End Model?

The End Model consists of three components that bookend the inference pipeline:

| Component | Purpose |
|-----------|---------|
| **Embedding Layer** | Converts text tokens into numerical vectors |
| **RMS Normalization** | Prepares hidden states for output projection |
| **Output Head** | Converts final hidden states back to token probabilities |

These components are deliberately grouped together because they're the only parts of the model that interact with human-readable content.

---

## Data Flow

### What Stays on the End Model Node

```
┌─────────────────────────────────────────────────────────────────┐
│                     END MODEL NODE                              │
│                                                                 │
│  User Prompt ──► Tokenizer ──► Token IDs ──► Embedding Layer   │
│  "Hello AI"      [15496, 9552]              [0.23, -0.14, ...]  │
│                                                                 │
│              ◄── Decoder ◄── Token IDs ◄── Output Head         │
│  "Hi there!"     [17250, 612]              [logits tensor]      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The End Model node handles:
- **Tokenization** — Converting your text prompt into numerical token IDs
- **Embedding** — Transforming token IDs into dense vector representations
- **Decoding** — Converting output token IDs back into readable text

**Your raw prompt text never leaves this machine.**

### What Gets Sent to Other Nodes

Other nodes in the network receive:
- **Hidden state tensors** — High-dimensional numerical arrays (e.g., 4096 floats per token)
- **Positional embeddings** — Mathematical representations of sequence position
- **Attention masks** — Binary masks indicating valid positions

These tensors are:
- **Unintelligible** without the embedding/head layers
- **Not reversible** to the original text
- **Mathematically transformed** representations with no human-readable content

---

## What Each Node Type Knows

### End Model Node

✓ The full text of your prompts  
✓ The full text of model responses  
✓ Token IDs and vocabulary mappings  
✓ Complete conversation history  

### Layer Nodes

✗ Cannot see the original prompt text  
✗ Cannot see the generated response text  
✗ Cannot reverse hidden states to tokens  
✓ Only process numerical tensor operations  

---

## Deployment Patterns

### Pattern 1: Maximum Privacy

Host the End Model yourself and distribute only the transformer layers:

```
┌─────────────────┐
│   Your Machine  │ ◄── You control the End Model
│   (End Model)   │     Your prompts stay here
│   Layers 0-10   │
└────────┬────────┘
         │ Hidden states only
         ▼
┌─────────────────┐
│  Friend's GPU   │ ◄── They help with compute
│   Layers 11-31  │     They never see your prompts
└─────────────────┘
```

Configuration for your machine:
```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 2
load_ends = true  # ← This enables the End Model
```

Configuration for the layer node:
```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cuda:0"
max_memory = 8
load_ends = false  # ← No access to prompts
```

### Pattern 2: Compute Contributor

Contribute your GPU to a network without ever seeing user data:

```toml
[[hosted_models]]
id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda:0"
max_memory = 16
load_ends = false  # ← You only process tensors
```

You provide compute power while remaining isolated from prompt content.

---

## Technical Details

### Why Hidden States Are Private

Hidden states are the intermediate representations within a transformer model. After the embedding layer transforms tokens into vectors, these vectors undergo dozens of attention and feed-forward operations.

**Key properties:**

1. **High dimensionality** — Each token becomes thousands of floating-point values
2. **Non-linear transformations** — Multiple layers of complex mathematical operations
3. **Context mixing** — Attention mechanisms blend information across positions
4. **No inverse function** — There's no mathematical way to recover tokens from hidden states without the output head

### The Embedding/Head Separation

The embedding layer and output head are mathematically linked:
- They often share weight matrices (tied embeddings)
- The head projects from hidden space → vocabulary space
- Without the head, you cannot determine which token a hidden state represents

This is why Language Pipes groups them together as the "End Model"—separating them would break the privacy guarantee.

---

## Security Considerations

### End Model Node Security

Since the End Model node has access to all prompts:
- Use disk encryption on the End Model machine
- Secure physical access to the device
- Enable `ecdsa_verification` to prevent unauthorized job injection

### Network Security

Enable encrypted communication between nodes:
```toml
network_key = "path/to/network.key"
```

This ensures hidden states are encrypted in transit.

---

## FAQ

### Can layer nodes reconstruct my prompts?

No. Without the embedding layer weights, there's no mapping from hidden states to tokens. The hidden states are dense numerical vectors with no human-interpretable meaning.

### What if someone captures the hidden states?

Hidden states in isolation are not useful. They require the exact model architecture, embedding/head weights, and knowledge of which layer produced them. Even with all of this, reversing the transformer operations is computationally intractable.

### Is this the same as encryption?

No—this is **architectural privacy**, not cryptographic privacy. The protection comes from the mathematical properties of transformer hidden states and the separation of model components. For additional security, enable the `network_key` option to encrypt all inter-node communication.

---

## See Also

- [Architecture Overview](./architecture.md) — How distributed inference works
- [Configuration Reference](./configuration.md) — The `load_ends` option
- [Interactive Setup](./interactive-setup.md) — Setting up the End Model
