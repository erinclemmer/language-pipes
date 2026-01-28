import json
import time
from typing import Callable, List

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.jobs.job import Job
from language_pipes.util.chat import ChatMessage
from language_pipes.util.http import _respond_json, _send_sse_headers

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
        max_completion_tokens = data['max_completion_tokens'] if 'max_completion_tokens' in data else 1000
        stream = data['stream'] if 'stream' in data else False
        temperature = data['temperature'] if 'temperature' in data else 1.0
        top_k = data['top_k'] if 'top_k' in data else 0
        top_p = data['top_p'] if 'top_p' in data else 1.0
        min_p = data['min_p'] if 'min_p' in data else 0.0
        presence_penalty = data['presence_penalty'] if 'presence_penalty' in data else 0.0
        return ChatCompletionRequest(data['model'], stream, max_completion_tokens, [ChatMessage.from_dict(m) for m in data['messages']], temperature, top_k, top_p, min_p, presence_penalty)

def send_initial_chunk(
    job: Job,
    created: float,
    handler: BaseHTTPRequestHandler
):
    msg = {
        "id": job.job_id,
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": job.model_id,
        "choices": [
            {
                "index": 0,
                "delta": { },
                "finish_reason": None
            }
        ]
    }
    data_bytes = json.dumps(msg).encode('utf-8')
    handler.wfile.write(b'event: response.creatted\ndata: ' + data_bytes + b'\n\n')
    handler.wfile.flush()

def send_update_chunk(
    job: Job,
    delta: object,
    created: float,
    finish_reason: str | None,
    handler: BaseHTTPRequestHandler
):
    msg = {
        "id": job.job_id,
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": job.model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
    data_bytes = json.dumps(msg).encode('utf-8')
    try:
        handler.wfile.write(b'data: ' + data_bytes + b'\n\n')
        handler.wfile.flush()
    except BrokenPipeError as e:
        print(e)
        return False # Stop job when pipe is broken
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
            "role": "assistant",
            "content": job.delta
        }, created_at, None, handler)
        
    def complete(job: Job):
        if type(job) == type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) == type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        else:
            if req.stream:
                send_update_chunk(job, { }, created_at, "stop", handler)
            else:
                _respond_json(handler, {
                    "id": job.job_id,
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
