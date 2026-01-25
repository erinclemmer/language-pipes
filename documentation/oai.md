# OpenAI-Compatible API

Language Pipes provides an OpenAI-compatible API server, allowing you to use existing tools and libraries designed for OpenAI's API.

> **Supported Endpoints:** Currently only `chat.completions` is supported. Other OpenAI endpoints are not yet implemented.

## Enabling the API Server

Set `oai_port` in your configuration to enable the API server:

```toml
oai_port = 8000
```

Or via CLI:
```bash
language-pipes serve --openai-port 8000 ...
```

---

## Using the OpenAI Python Library

Language Pipes is fully compatible with the [OpenAI Python library](https://github.com/openai/openai-python).

```bash
pip install openai
```

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Language Pipes doesn't require authentication
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is distributed computing?"}
    ],
    max_completion_tokens=200
)

print(response.choices[0].message.content)
```

### Streaming Responses

For real-time token-by-token output, use `stream=True`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[
        {"role": "user", "content": "Write a short poem about networks."}
    ],
    max_completion_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # Newline at end
```

## Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_completion_tokens": 50
  }'
```

### Streaming with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_completion_tokens": 50,
    "stream": true
  }'
```

---

## API Reference

### Endpoint

```
POST /v1/chat/completions
```

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `model` | string | ✓ | Model ID (must match a hosted model) |
| `messages` | array | ✓ | Array of message objects |
| `max_completion_tokens` | integer | | Maximum tokens to generate (default: 1000) |
| `stream` | boolean | | Enable streaming responses (default: `false`) |
| `temperature` | float | | Controls output randomness (default: `1.0`) |
| `top_p` | float | | Nucleus sampling threshold (default: `1.0`) |
| `top_k` | integer | | Top-k sampling limit (default: `0`, disabled) |
| `min_p` | float | | Minimum probability threshold (default: `0`, disabled) |
| `presence_penalty` | float | | Penalty for token repetition (default: `0`) |

### Sampling Parameters

Language Pipes supports several sampling parameters to control text generation. When `temperature > 0`, these parameters are applied in the following order: temperature scaling → min_p filtering → top_p filtering → top_k filtering.

#### Temperature

The `temperature` parameter controls output randomness by scaling logits before softmax:

```
scaled_logits = logits / temperature
probabilities = softmax(scaled_logits)
```

- **`temperature = 0`** → Greedy decoding (always picks the most likely token)
- **`temperature < 1`** → Sharper distribution, more deterministic output
- **`temperature = 1`** → Standard softmax (no scaling)
- **`temperature > 1`** → Flatter distribution, more random/creative output

Lower temperatures make the model more confident and focused on likely tokens. Higher temperatures increase diversity by giving more probability mass to less likely tokens.

#### Top-p (Nucleus Sampling)

The `top_p` parameter implements nucleus sampling, which limits token selection to the smallest set of tokens whose cumulative probability exceeds the threshold:

- **`top_p = 1.0`** → Disabled (consider all tokens)
- **`top_p = 0.9`** → Only sample from tokens comprising the top 90% probability mass
- **`top_p = 0.5`** → Only sample from tokens comprising the top 50% probability mass

Lower values make output more focused by excluding low-probability tokens from consideration.

#### Top-k

The `top_k` parameter limits sampling to the k most likely tokens:

- **`top_k = 0`** → Disabled (consider all tokens)
- **`top_k = 50`** → Only sample from the 50 most likely tokens
- **`top_k = 1`** → Equivalent to greedy decoding

This provides a hard cutoff on the number of tokens considered, regardless of their probability distribution.

#### Min-p

The `min_p` parameter filters out tokens whose probability is below a fraction of the most likely token's probability:

```
threshold = min_p * max_probability
```

- **`min_p = 0`** → Disabled (consider all tokens)
- **`min_p = 0.1`** → Remove tokens with probability < 10% of the top token's probability
- **`min_p = 0.05`** → Remove tokens with probability < 5% of the top token's probability

This provides adaptive filtering that scales with the model's confidence—when the model is very confident, fewer tokens pass the threshold.

#### Presence Penalty

The `presence_penalty` parameter discourages the model from repeating tokens that have already appeared in the generation:

```
logits[token] -= presence_penalty  (for each token that has appeared)
```

- **`presence_penalty = 0`** → Disabled (no penalty)
- **`presence_penalty > 0`** → Reduce probability of repeated tokens
- **`presence_penalty < 0`** → Increase probability of repeated tokens (encourage repetition)

Unlike frequency penalty, presence penalty applies equally to all tokens that have appeared, regardless of how many times they occurred.

### Message Object

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | `system`, `user`, or `assistant` |
| `content` | string | Message content |

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-1.7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

---

## Notes

- **No API key required** — Language Pipes does not implement authentication. Any value works for `api_key`.
- **Model names** — Use the exact HuggingFace model ID you configured (e.g., `Qwen/Qwen3-1.7B`)
- **Network access** — Ensure the client can reach the node hosting the OpenAI server

---

## See Also

- [Configuration](./configuration.md) — Enable the API with `oai_port`
- [CLI Reference](./cli.md) — Command-line usage
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) — Official documentation
