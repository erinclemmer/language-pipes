# Privacy Architecture

Language Pipes provides privacy-preserving distributed inference through its **End Model** architecture. This document explains how your prompt data stays protected when using a distributed network.

---

## How Language Models Work (Quick Background)

To understand why the End Model architecture protects your privacy, it helps to know how language models process text:

1. **Models don't understand text directly.** They only work with numbers. Every word or piece of a word must be converted to numbers before the model can process it.

2. **Models are built in layers.** A typical model has 30-80 transformer layers stacked on top of each other. Each layer takes numbers in, does mathematical operations, and passes numbers to the next layer.

3. **Only the first and last parts deal with text.** The first part (embedding) converts your text into numbers. The last part (output head) converts numbers back into text. Everything in between just shuffles numbers around.

This is why Language Pipes can split models across computers while keeping your data private: **only the computer with the first and last parts can read your text.**

## The End Model Concept


In Language Pipes, only the node hosting the **End Model** ever sees your actual prompts and responses. All other nodes process numbers that are meaningless without the End Model.

### What is the End Model?

The End Model groups the components that convert between text and numbers:

| Component | What It Does |
|-----------|--------------|
| **Embedding Layer** | Converts words → numbers (the "on-ramp" to the model) |
| **RMS Normalization** | Prepares the final numbers for output |
| **Output Head** | Converts numbers → words (the "off-ramp" from the model) |

Think of it like a translation system: the embedding layer translates English into a secret language of numbers, and the output head translates that secret language back to English. The middle layers only speak the secret number language—they never see English at all.

### Why can’t layer nodes reverse-engineer the prompt?

Layer activations are **not a unique or lossless encoding of the input prompt**. While they contain information about the prompt, they are not designed to preserve it in a form that can be perfectly reversed.

In idealized theory, exact inversion is sometimes possible under strong assumptions (real-valued arithmetic, full access to all hidden states, and no noise). **Real systems break these assumptions**:

* **Information is mixed and normalized.** Operations like attention and normalization deliberately blend token information rather than store it verbatim.
* **Precision is limited.** Activations are stored and transmitted with finite precision, which causes different prompts to collapse to indistinguishable internal states.
* **Observations are approximate.** Any practical comparison of activations relies on tolerances, which guarantees collisions.
* **Single layers are insufficient.** A single hidden state—or even a full layer snapshot—can correspond to multiple valid prompts.

As a result, while one might infer a *plausible* prompt statistically, **there is no general, reliable way for layer nodes to recover the exact original prompt**. This is a fundamental property of how modern language models represent information.

## Data Flow: Step by Step

### Step 1: You Enter a Prompt

You type a message to the AI. This raw text exists only on your machine.

```
"What is the capital of France?"
```

### Step 2: Tokenization

The tokenizer splits your text into pieces and looks up each piece in a vocabulary to get integer IDs. These token IDs could be decoded by anyone with the same vocabulary, so they stay local.

```
"What is the capital of France?"  →  [ 1841, 374, 279, 6864, 315, 9822, 30 ]
```

### Step 3: Embedding

The embedding layer converts each token ID into a vector of thousands of floating-point numbers. This is a one-way transformation—you can't reverse it without the embedding matrix weights.

```
Token IDs: [ 1841, 374, 279, ... ]
                    ↓
Hidden states: [[ 0.023, -0.147, 0.892, ... ],    ← 4096 floats per token
                [ 0.156,  0.089, -0.445, ... ],
                ...                           ]
```

**After this step, your text and token IDs never leave your machine.** Only the hidden state numbers proceed to other nodes.

### Step 4: Data Sent to Layer Nodes

**Only these numerical arrays leave your machine:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Hidden State Tensor: [1, 7, 4096]  ← 7 positions × 4096 floats each    │
│  Position IDs: [0, 1, 2, 3, 4, 5, 6]                                    │
│  Attention Mask: [[1, 1, 1, 1, 1, 1, 1]]                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ⚠️  NOT INCLUDED:                                                      │
│      • Original prompt                                                  │
│      • Token IDs                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```


### Step 5: Layer Processing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LAYER NODE                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  INPUT:  [1, 7, 4096] floats  →  Transformer Layers  →  OUTPUT: floats  │
├─────────────────────────────────────────────────────────────────────────┤
│  🔒 Knows: Numbers came in, matrix math happened, numbers went out      │
│  🚫 Cannot know: Language, words, topic, or any semantic meaning        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 6: Hidden States Return (End Model Node)

The processed hidden states come back from the layer nodes. These are still just arrays of floating-point numbers—no text or tokens.

```
Returned hidden states: [[ 0.891, -0.234, 0.567, ... ],    ← Still 4096 floats per token
                         [ 0.445, 0.123, -0.789, ... ],
                         ...                         ]
```

### Step 7: Output Head (End Model Node)

The output head converts the final hidden state into scores for every word in the vocabulary. Depending upon your request parameters a new token is chosen from these scores.


```
Hidden state: [ 0.891, -0.234, 0.567, ... ]
                         ↓
              Output Head (matrix multiplication)
                         ↓
Token ID: 12366  (chosen based on scores and request parameters)
```

### Step 8: Decoding (End Model Node)

The tokenizer looks up the token ID in the vocabulary to produce readable text. This text exists only on your machine.

```
Token ID: 12366  →  "Paris"
```

**The generated response never leaves your machine.** Only you see the final text output.
## Deployment Patterns

### Maximum Privacy: Host the End Model Yourself

```
┌─────────────────┐
│   Your Machine  │ ◄── End Model: prompts stay here
│   Layers 0-10   │
└────────┬────────┘
         │ Hidden states only
         ▼
┌─────────────────┐
│  Friend's GPU   │ ◄── Layers only: never sees prompts
│   Layers 11-31  │
└─────────────────┘
```

```toml
# Your machine
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
load_ends = true   # ← You control the End Model

# Friend's machine  
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
load_ends = false  # ← Does not matter for privacy
```

Your friend can choose to also host an end model for the model but only your machine will see your prompts if you send a request to your machine.

---


## FAQ

**Can layer nodes reconstruct my prompts?**  
No. Without embedding weights, hidden states cannot be mapped back to tokens.

**What if someone captures the hidden states?**  
Hidden states require the exact model weights and architecture to decode. Reversing transformer operations is computationally intractable.

**Is this encryption?**  
No—it's architectural privacy. Protection comes from model component separation, not cryptography. For additional AES encryption security, set `network_key` in the configuration.


## See Also

- [Architecture Overview](./architecture.md) — How distributed inference works
- [Configuration Reference](./configuration.md) — The `load_ends` option
