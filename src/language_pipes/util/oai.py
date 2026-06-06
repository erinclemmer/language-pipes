import json
import time
from typing import Any, Callable, List, Optional

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.jobs.job import Job
from language_pipes.util.chat import ChatMessage, ChatRole
from language_pipes.util.http import _respond_json, _send_sse_headers
from language_pipes.util.oai_chunks import send_complete, send_initial_chunk, send_update_chunk

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

        if item_type == "message" or role is not None:
            content = _content_to_text(item.get("content", ""))
            messages.append(ChatMessage.from_dict({
                "role": role or "user",
                "content": content
            }))
        elif item_type == "function_call_output":
            output = _content_to_text(item.get("output", ""))
            if output:
                messages.append(ChatMessage(ChatRole.USER, output))

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
            presence_penalty: float = 0.0
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
        messages = _response_input_to_messages(data['input'])
        if instructions is not None:
            messages.insert(0, ChatMessage(ChatRole.SYSTEM, str(instructions)))
        if len(messages) == 0:
            raise ValueError("input must contain at least one text message")

        return ResponsesRequest(data['model'], stream, data['input'], instructions, max_output_tokens, messages, temperature, top_k, top_p, min_p, presence_penalty)

def _response_json(job: Any, req: ResponsesRequest, created_at: float):
    output_text = job.result
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
        "output": [{
            "id": f"msg-{job.job_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": output_text,
                "annotations": []
            }]
        }],
        "output_text": output_text,
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

def oai_chat_complete(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict):
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
        complete_cb(req.model, req.messages, req.max_completion_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def oai_responses_create(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict):
    try:
        req = ResponsesRequest.from_dict(data)
    except ValueError as e:
        _respond_json(handler, { "error": str(e) })
        return

    created_at = time.time()
    output_item_id = None

    def start(job: Job):
        nonlocal output_item_id
        output_item_id = f"msg-{job.job_id}"
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
        if not _write_response_event(handler, "response.created", {"response": response}):
            return False
        return _write_response_event(handler, "response.output_item.added", {
            "output_index": 0,
            "item": {
                "id": output_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        })

    def update(job: Job):
        if not req.stream:
            return True
        return _write_response_event(handler, "response.output_text.delta", {
            "item_id": output_item_id or f"msg-{job.job_id}",
            "output_index": 0,
            "content_index": 0,
            "delta": job.delta
        })

    def complete(job: Job):
        if type(job) is type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) is type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        else:
            response = _response_json(job, req, created_at)
            if req.stream:
                _write_response_event(handler, "response.output_text.done", {
                    "item_id": output_item_id or f"msg-{job.job_id}",
                    "output_index": 0,
                    "content_index": 0,
                    "text": job.result
                })
                _write_response_event(handler, "response.output_item.done", {
                    "output_index": 0,
                    "item": response["output"][0]
                })
                _write_response_event(handler, "response.completed", {"response": response})
                try:
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                except Exception:
                    pass
            else:
                _respond_json(handler, response)

    def promise_fn(resolve: Callable, _: Callable):
        complete_cb(req.model, req.messages, req.max_output_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def get_models(handler: BaseHTTPRequestHandler, get_models: Callable):
    models = get_models()
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
