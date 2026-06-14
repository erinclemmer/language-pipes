# OpenAI-Compatible API

Language Pipes provides an OpenAI-compatible API server, allowing you to use existing tools and libraries designed for OpenAI's API.

> **Supported Endpoints:** `chat.completions`, `responses`, and `models` are supported. Other OpenAI endpoints are not yet implemented.

## Enabling the API Server

Set `job_port` in your configuration to enable the API server:

```toml
job_port = 8000
```

Or via CLI:
```bash
language-pipes serve --openai-port 8000 ...
```

Optionally you can set an API key for use with the server. Any number of api keys will work with the flag:

```toml
api_keys = ["foo", "bar", "baz"]
```

Or via CLI:
```bash
language-pipes serve --openai-port 8000 --api-keys foo bar baz
```

---

## Using the OpenAI Python Library

Language Pipes is fully compatible with the [OpenAI Python library](https://github.com/openai/openai-python).

```bash
pip install openai
```

### Basic Usage

Run the serve like this:
```bash
language-pipes serve --openai-port 8000 --api-keys foo
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="foo"
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

### Responses API Usage

The `/v1/responses` endpoint supports the newer OpenAI Responses API shape for text generation. Use `instructions` for system-level guidance and `input` for a string prompt or compatible message items.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="foo"
)

response = client.responses.create(
    model="Qwen/Qwen3-1.7B",
    instructions="You are a helpful assistant.",
    input="What is distributed computing?",
    max_output_tokens=200
)

print(response.output_text)
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

### Function Tool Calling

The `/v1/responses` endpoint supports OpenAI Responses **custom function tools**. Pass tool definitions in `tools`; when the model decides to call a tool, the response contains a `function_call` output item instead of a text message. Your application executes the function and sends the result back as a `function_call_output` input item.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="foo"
)

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

response = client.responses.create(
    model="Qwen/Qwen3-1.7B",
    input="What is the weather in Chicago?",
    tools=tools
)

# When a tool is called, response.output contains a function_call item:
for item in response.output:
    if item.type == "function_call":
        print(item.name, item.arguments)  # get_weather {"city": "Chicago"}
```

Send the tool result back by including the prior `function_call` item and a matching `function_call_output` linked by `call_id`:

```python
response = client.responses.create(
    model="Qwen/Qwen3-1.7B",
    input=[
        {"role": "user", "content": "What is the weather in Chicago?"},
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": "{\"city\": \"Chicago\"}"
        },
        {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "{\"temperature\": 72}"
        }
    ],
    tools=tools
)
```

#### Tool Calling Limitations

- **Custom function tools only.** Hosted tools (`web_search`, `file_search`, `computer_use`, `code_interpreter`, MCP) are rejected with a `400`.
- **No server-side execution.** Tool calls are returned for your client to execute; Language Pipes never runs the function.
- **Quality depends on the model.** Tools are injected as a model-agnostic instruction block asking the model to emit a JSON tool call. Reliability varies with the model's instruction-following ability, and smaller models may emit malformed JSON that is treated as plain text.
- **Reasoning models.** A leading `<think>…</think>` block is split off and returned as a separate `reasoning` output item (its text under `summary[0].text`); `output_text` and the `message` item contain only the answer. The tool-call parser is likewise tolerant of a reasoning block and of prose surrounding the JSON. When streaming, reasoning is streamed live token-by-token to the reasoning item (the `<think>`/`</think>` markers are stripped) and the item is closed before the answer begins — see the streaming note below.
- **Single tool call per response.** Parallel tool calls are not produced even when `parallel_tool_calls` is set.
- **Buffered streaming for tools.** When `tools` are supplied, `stream=true` cannot stream live token deltas because the output cannot be classified as text vs. a tool call until generation finishes. Every output item (including any `reasoning` item) is emitted at completion: for a tool call, `response.output_item.added` (a `function_call` item) → `response.function_call_arguments.delta` (the full arguments in one delta) → `response.function_call_arguments.done` → `response.output_item.done` → `response.completed`. Requests without `tools` stream token-by-token.
- **Reasoning streaming.** With no tools and `stream=true`, a leading `<think>…</think>` block streams live: `response.output_item.added` (a `reasoning` item) → one or more `response.reasoning_summary_text.delta` events (the `<think>`/`</think>` markers stripped, with a bounded lookahead so a tag split across token deltas is never leaked) → `response.reasoning_summary_text.done` → `response.output_item.done`. The answer that follows then streams as normal `response.output_text.delta` events on a `message` item at the next `output_index`.

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

### Responses API with curl

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "instructions": "You are a helpful assistant.",
    "input": "Hello!",
    "max_output_tokens": 50
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
POST /v1/responses
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

### Responses Request Body

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `model` | string | ✓ | Model ID (must match a hosted model) |
| `input` | string or array | ✓ | Prompt text or compatible Responses message items |
| `instructions` | string | | System-level guidance prepended to the prompt |
| `max_output_tokens` | integer | | Maximum tokens to generate (default: 1000) |
| `stream` | boolean | | Enable typed server-sent event responses (default: `false`) |
| `temperature` | float | | Controls output randomness (default: `1.0`) |
| `top_p` | float | | Nucleus sampling threshold (default: `1.0`) |
| `top_k` | integer | | Top-k sampling limit (default: `0`, disabled) |
| `min_p` | float | | Minimum probability threshold (default: `0`, disabled) |
| `presence_penalty` | float | | Penalty for token repetition (default: `0`) |
| `tools` | array | | Custom function tool definitions (see [Function Tool Calling](#function-tool-calling)) |
| `tool_choice` | string or object | | `auto`, `none`, `required`, or `{"type": "function", "name": "..."}` |
| `parallel_tool_calls` | boolean | | Accepted for compatibility; parallel calls are not produced |

The endpoint returns a Responses API-style object with `output`, `output_text`, and `usage` fields. Custom function tools are supported; hosted tools, `previous_response_id` statefulness, and multimodal input are not currently implemented.

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

### Documentation
* [CLI Reference](./cli.md)
* [Privacy Protection](./privacy.md)
* [Configuration Manual](./configuration.md)
* [Architecture Overview](./architecture.md)
* [Open AI Compatable API](./oai.md)
* [Job Processor State Machine](./job-processor.md)
* [The default peer to peer implementation](./distributed-state-network/README.md)
* [The way Language Pipes abstracts from model architecture](./llm-layer-collector.md)