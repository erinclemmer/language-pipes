import json
import time
from typing import Any, Callable, List, Optional

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.jobs.job import Job
from language_pipes.util.chat import ChatMessage, ChatRole
from language_pipes.util.http import _respond_json, _send_code, _send_sse_headers
from language_pipes.util.oai_chunks import send_complete, send_initial_chunk, send_update_chunk
from language_pipes.util.oai_tool_calls import (
    ReasoningStreamSplitter,
    ResponsesTool,
    build_tool_instructions,
    format_assistant_tool_call,
    format_tool_result,
    parse_tool_call,
    parse_tool_definitions,
    split_reasoning,
    validate_tool_choice,
)

class ChatCompletionRequest:
    model: str
    stream: bool
    messages: List[ChatMessage]
    max_completion_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float

    def __init__(
            self, 
            model: str, 
            stream: bool,
            max_completion_tokens: int,
            messages: List[ChatMessage],
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0
        ):
        self.model = model
        self.stream = stream
        self.max_completion_tokens = max_completion_tokens
        self.messages = messages
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty

    def to_json(self):
        return {
            'model': self.model,
            'stream': self.stream,
            'max_completion_tokens': self.max_completion_tokens,
            'messages': [m.to_json() for m in self.messages],
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_p': self.min_p,
            'presence_penalty': self.presence_penalty
        }
    
    @staticmethod
    def from_dict(data):
        max_completion_tokens = 1000
        if "max_tokens" in data:
            max_completion_tokens = data['max_tokens']
        if "max_completion_tokens" in data:
            max_completion_tokens = data['max_completion_tokens']
        
        stream = data['stream'] if 'stream' in data else False
        temperature = data['temperature'] if 'temperature' in data else 1.0
        top_k = data['top_k'] if 'top_k' in data else 0
        top_p = data['top_p'] if 'top_p' in data else 1.0
        min_p = data['min_p'] if 'min_p' in data else 0.0
        presence_penalty = data['presence_penalty'] if 'presence_penalty' in data else 0.0
        return ChatCompletionRequest(data['model'], stream, max_completion_tokens, [ChatMessage.from_dict(m) for m in data['messages']], temperature, top_k, top_p, min_p, presence_penalty)

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)

def _response_input_to_messages(response_input: Any) -> List[ChatMessage]:
    if isinstance(response_input, str):
        return [ChatMessage(ChatRole.USER, response_input)]

    items = response_input if isinstance(response_input, list) else [response_input]
    messages = []
    for item in items:
        if isinstance(item, str):
            messages.append(ChatMessage(ChatRole.USER, item))
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "function_call":
            # A prior assistant tool call. Replay it as an assistant message so
            # the model has the context of what it asked for before it sees the
            # corresponding tool result.
            messages.append(ChatMessage(
                ChatRole.ASSISTANT,
                format_assistant_tool_call(
                    item.get("name"),
                    item.get("arguments", ""),
                    item.get("call_id")
                )
            ))
        elif item_type == "function_call_output":
            output = _content_to_text(item.get("output", ""))
            messages.append(ChatMessage(
                ChatRole.USER,
                format_tool_result(item.get("call_id"), output)
            ))
        elif item_type == "message" or role is not None:
            content = _content_to_text(item.get("content", ""))
            messages.append(ChatMessage.from_dict({
                "role": role or "user",
                "content": content
            }))

    return messages

class ResponsesRequest:
    model: str
    stream: bool
    input: Any
    instructions: Optional[str]
    messages: List[ChatMessage]
    max_output_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    tools: List[ResponsesTool]
    tool_choice: Any
    parallel_tool_calls: bool

    def __init__(
            self,
            model: str,
            stream: bool,
            response_input: Any,
            instructions: Optional[str],
            max_output_tokens: int,
            messages: List[ChatMessage],
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            tools: Optional[List[ResponsesTool]] = None,
            tool_choice: Any = None,
            parallel_tool_calls: bool = False
        ):
        self.model = model
        self.stream = stream
        self.input = response_input
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens
        self.messages = messages
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.tools = tools if tools is not None else []
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

    @staticmethod
    def from_dict(data):
        max_output_tokens = 1000
        if "max_tokens" in data:
            max_output_tokens = data['max_tokens']
        if "max_completion_tokens" in data:
            max_output_tokens = data['max_completion_tokens']
        if "max_output_tokens" in data:
            max_output_tokens = data['max_output_tokens']

        stream = data['stream'] if 'stream' in data else False
        temperature = data['temperature'] if 'temperature' in data else 1.0
        top_k = data['top_k'] if 'top_k' in data else 0
        top_p = data['top_p'] if 'top_p' in data else 1.0
        min_p = data['min_p'] if 'min_p' in data else 0.0
        presence_penalty = data['presence_penalty'] if 'presence_penalty' in data else 0.0
        instructions = data.get('instructions')

        tools: List[ResponsesTool] = []
        tool_choice = data.get('tool_choice')
        parallel_tool_calls = bool(data.get('parallel_tool_calls', False))
        if data.get('tools') is not None:
            tools = parse_tool_definitions(data['tools'])
            validate_tool_choice(tool_choice, tools)

        messages = _response_input_to_messages(data['input'])
        if instructions is not None:
            messages.insert(0, ChatMessage(ChatRole.SYSTEM, str(instructions)))
        if len(tools) > 0:
            # Inject tool schemas as a system-level instruction so the model
            # sees the available tools and the expected JSON output format.
            tool_message = ChatMessage(ChatRole.SYSTEM, build_tool_instructions(tools, tool_choice))
            insert_at = 1 if instructions is not None else 0
            messages.insert(insert_at, tool_message)
        if len(messages) == 0:
            raise ValueError("input must contain at least one text message")

        return ResponsesRequest(data['model'], stream, data['input'], instructions, max_output_tokens, messages, temperature, top_k, top_p, min_p, presence_penalty, tools, tool_choice, parallel_tool_calls)

def _reasoning_item(job: Any, reasoning_text: str) -> dict:
    return {
        "id": f"rs-{job.job_id}",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": reasoning_text}]
    }

def _response_json(job: Any, req: ResponsesRequest, created_at: float):
    reasoning_text, content = split_reasoning(job.result)
    tool_call = parse_tool_call(job.result, req.tools)

    response_output = []
    if reasoning_text:
        response_output.append(_reasoning_item(job, reasoning_text))

    if tool_call is not None:
        response_output.append({
            "id": f"fc-{job.job_id}",
            "type": "function_call",
            "status": "completed",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": tool_call.arguments
        })
        response_output_text = ""
    else:
        response_output.append({
            "id": f"msg-{job.job_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": []
            }]
        })
        response_output_text = content

    return {
        "id": f"resp-{job.job_id}",
        "object": "response",
        "created_at": int(created_at),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": req.instructions,
        "max_output_tokens": req.max_output_tokens,
        "model": job.model_id,
        "output": response_output,
        "output_text": response_output_text,
        "usage": {
            "input_tokens": job.prompt_tokens,
            "output_tokens": job.current_token,
            "total_tokens": job.prompt_tokens + job.current_token
        }
    }

def _write_response_event(handler: BaseHTTPRequestHandler, event_type: str, data: dict):
    msg = {"type": event_type}
    msg.update(data)
    try:
        handler.wfile.write(b"data: " + json.dumps(msg).encode("utf-8") + b"\n\n")
        handler.wfile.flush()
    except Exception:
        return False
    return True

def oai_chat_complete(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict, api_key: str):
    req = ChatCompletionRequest.from_dict(data)
    created_at = time.time()

    def start(job: Job):
        if not req.stream:
            return
        _send_sse_headers(handler)
        send_initial_chunk(job, created_at, handler)

    def update(job: Job):
        if not req.stream:
            return True
        return send_update_chunk(job, {
            "content": job.delta
        }, created_at, None, handler)
        
    def complete(job: Job):
        if type(job) is type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) is type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        else:
            if req.stream:
                send_complete(job, created_at, handler)
            else:
                _respond_json(handler, {
                    "id": f"chatcmpl-{job.job_id}",
                    "object": "chat.completion",
                    "created": int(created_at),
                    "model": job.model_id,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": job.result
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": job.prompt_tokens,
                        "completion_tokens": job.current_token,
                        "total_tokens": job.prompt_tokens + job.current_token
                    }
                })

    def promise_fn(resolve: Callable, _: Callable):
        complete_cb(api_key, req.model, req.messages, req.max_completion_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def oai_responses_create(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict, api_key: str):
    try:
        req = ResponsesRequest.from_dict(data)
    except ValueError as e:
        _send_code(400, handler, str(e))
        return

    created_at = time.time()
    # When tools are present the output cannot be classified as text vs. a tool
    # call until generation finishes, so we buffer (suppress live deltas) and
    # emit every item at completion. Plain text requests stream token-by-token;
    # a leading <think> reasoning block streams live to a reasoning item, which
    # is closed when the answer begins.
    buffering = len(req.tools) > 0
    splitter = ReasoningStreamSplitter()
    # Live (non-buffered) streaming state, mutated across start/update/complete.
    sstate = {
        "reasoning_id": None,
        "message_id": None,
        "reasoning_index": None,
        "message_index": None,
        "next_index": 0,
        "reasoning_acc": "",
        "reasoning_open": False,
        "reasoning_closed": False,
        "message_open": False,
        "content_acc": "",
    }

    def _open_reasoning():
        sstate["reasoning_index"] = sstate["next_index"]
        sstate["next_index"] += 1
        sstate["reasoning_open"] = True
        _write_response_event(handler, "response.output_item.added", {
            "output_index": sstate["reasoning_index"],
            "item": {"id": sstate["reasoning_id"], "type": "reasoning", "status": "in_progress", "summary": []}
        })

    def _reasoning_delta(delta: str):
        # Stream reasoning tokens live to the reasoning item.
        if not sstate["reasoning_open"]:
            _open_reasoning()
        sstate["reasoning_acc"] += delta
        return _write_response_event(handler, "response.reasoning_summary_text.delta", {
            "item_id": sstate["reasoning_id"],
            "output_index": sstate["reasoning_index"],
            "summary_index": 0,
            "delta": delta
        })

    def _close_reasoning():
        if not sstate["reasoning_open"] or sstate["reasoning_closed"]:
            return
        sstate["reasoning_closed"] = True
        text = sstate["reasoning_acc"]
        _write_response_event(handler, "response.reasoning_summary_text.done", {
            "item_id": sstate["reasoning_id"], "output_index": sstate["reasoning_index"],
            "summary_index": 0, "text": text
        })
        _write_response_event(handler, "response.output_item.done", {
            "output_index": sstate["reasoning_index"],
            "item": {
                "id": sstate["reasoning_id"], "type": "reasoning", "status": "completed",
                "summary": [{"type": "summary_text", "text": text}] if text else []
            }
        })

    def _open_message():
        sstate["message_index"] = sstate["next_index"]
        sstate["next_index"] += 1
        sstate["message_open"] = True
        _write_response_event(handler, "response.output_item.added", {
            "output_index": sstate["message_index"],
            "item": {
                "id": sstate["message_id"], "type": "message", "status": "in_progress",
                "role": "assistant", "content": []
            }
        })

    def _content_delta(delta: str):
        # The reasoning block (if any) is fully streamed before the answer; close
        # it before the first content token.
        if sstate["reasoning_open"] and not sstate["reasoning_closed"]:
            _close_reasoning()
        if not sstate["message_open"]:
            _open_message()
        sstate["content_acc"] += delta
        return _write_response_event(handler, "response.output_text.delta", {
            "item_id": sstate["message_id"],
            "output_index": sstate["message_index"],
            "content_index": 0,
            "delta": delta
        })

    def start(job: Job):
        sstate["reasoning_id"] = f"rs-{job.job_id}"
        sstate["message_id"] = f"msg-{job.job_id}"
        if not req.stream:
            return
        _send_sse_headers(handler)
        response = {
            "id": f"resp-{job.job_id}",
            "object": "response",
            "created_at": int(created_at),
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "instructions": req.instructions,
            "max_output_tokens": req.max_output_tokens,
            "model": job.model_id,
            "output": [],
            "output_text": ""
        }
        # output_item.added is deferred until the output type/ordering is known
        # (a reasoning item may precede the message/function_call item).
        return _write_response_event(handler, "response.created", {"response": response})

    def update(job: Job):
        if not req.stream or buffering:
            return True
        reasoning_delta, content_delta = splitter.feed(job.delta)
        ok = True
        if reasoning_delta:
            ok = _reasoning_delta(reasoning_delta) and ok
        if content_delta:
            ok = _content_delta(content_delta) and ok
        return ok

    def _complete_buffered(response: dict):
        # Nothing has streamed yet; emit each output item in full.
        for index, item in enumerate(response["output"]):
            item_type = item["type"]
            if item_type == "reasoning":
                text = item["summary"][0]["text"] if item["summary"] else ""
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {"id": item["id"], "type": "reasoning", "status": "in_progress", "summary": []}
                })
                _write_response_event(handler, "response.reasoning_summary_text.delta", {
                    "item_id": item["id"], "output_index": index, "summary_index": 0, "delta": text
                })
                _write_response_event(handler, "response.reasoning_summary_text.done", {
                    "item_id": item["id"], "output_index": index, "summary_index": 0, "text": text
                })
            elif item_type == "function_call":
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {
                        "id": item["id"], "type": "function_call", "status": "in_progress",
                        "call_id": item["call_id"], "name": item["name"], "arguments": ""
                    }
                })
                _write_response_event(handler, "response.function_call_arguments.delta", {
                    "item_id": item["id"], "output_index": index, "delta": item["arguments"]
                })
                _write_response_event(handler, "response.function_call_arguments.done", {
                    "item_id": item["id"], "output_index": index, "arguments": item["arguments"]
                })
            else:  # message
                text = item["content"][0]["text"]
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {
                        "id": item["id"], "type": "message", "status": "in_progress",
                        "role": "assistant", "content": []
                    }
                })
                _write_response_event(handler, "response.output_text.delta", {
                    "item_id": item["id"], "output_index": index, "content_index": 0, "delta": text
                })
                _write_response_event(handler, "response.output_text.done", {
                    "item_id": item["id"], "output_index": index, "content_index": 0, "text": text
                })
            _write_response_event(handler, "response.output_item.done", {
                "output_index": index, "item": item
            })

    def _complete_live(job: Job, response: dict):
        # Flush any held-back lookahead, then reconcile against the authoritative
        # split of the full result in case deltas did not sum to job.result.
        flush_reasoning, flush_content = splitter.finalize()
        if flush_reasoning:
            _reasoning_delta(flush_reasoning)
        if flush_content:
            _content_delta(flush_content)

        final_reasoning, final_content = split_reasoning(job.result)
        # Reconcile reasoning, then close it before any content (in case deltas
        # did not stream it, e.g. when the pipeline delivered no live updates).
        if final_reasoning:
            r_remainder = final_reasoning[len(sstate["reasoning_acc"]):] \
                if final_reasoning.startswith(sstate["reasoning_acc"]) else final_reasoning
            if r_remainder:
                _reasoning_delta(r_remainder)
        if sstate["reasoning_open"] and not sstate["reasoning_closed"]:
            _close_reasoning()

        c_remainder = final_content[len(sstate["content_acc"]):] \
            if final_content.startswith(sstate["content_acc"]) else final_content
        if c_remainder:
            _content_delta(c_remainder)
        if not sstate["message_open"]:
            _open_message()

        message_item = next(i for i in response["output"] if i["type"] == "message")
        _write_response_event(handler, "response.output_text.done", {
            "item_id": sstate["message_id"],
            "output_index": sstate["message_index"],
            "content_index": 0,
            "text": final_content
        })
        _write_response_event(handler, "response.output_item.done", {
            "output_index": sstate["message_index"], "item": message_item
        })

    def complete_stream(job: Job, response: dict):
        if buffering:
            _complete_buffered(response)
        else:
            _complete_live(job, response)
        _write_response_event(handler, "response.completed", {"response": response})
        try:
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except Exception:
            pass

    def complete(job: Job):
        if type(job) is type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) is type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        else:
            response = _response_json(job, req, created_at)
            if req.stream:
                complete_stream(job, response)
            else:
                _respond_json(handler, response)

    def promise_fn(resolve: Callable, _: Callable):
        complete_cb(api_key, req.model, req.messages, req.max_output_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def get_models(handler: BaseHTTPRequestHandler, get_models: Callable):
    models = get_models()
    try:
        _respond_json(handler, {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": ""
                } for m in models
            ]
        })
    except:  # noqa: E722
        pass 
