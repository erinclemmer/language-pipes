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

### Async Usage

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )

    response = await client.chat.completions.create(
        model="Qwen/Qwen3-1.7B",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        max_completion_tokens=50
    )

    print(response.choices[0].message.content)

asyncio.run(main())
```

### Async Streaming

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )

    stream = await client.chat.completions.create(
        model="Qwen/Qwen3-1.7B",
        messages=[
            {"role": "user", "content": "Count to 10."}
        ],
        max_completion_tokens=100,
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()

asyncio.run(main())
```

---

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
| `max_completion_tokens` | integer | | Maximum tokens to generate |
| `stream` | boolean | | Enable streaming responses (default: `false`) |

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
