import json
import time
from typing import Callable, List

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.util.chat import ChatMessage
from language_pipes.util.http import _respond_json, _send_sse_headers
from language_pipes.job_manager.job import Job

class ChatCompletionRequest:
    model: str
    stream: bool
    messages: List[ChatMessage]
    max_completion_tokens: int

    def __init__(
            self, 
            model: str, 
            stream: bool,
            max_completion_tokens: int,
            messages: List[ChatMessage]
        ):
        self.model = model
        self.stream = stream
        self.max_completion_tokens = max_completion_tokens
        self.messages = messages

    def to_json(self):
        return {
            'model': self.model,
            'stream': self.stream,
            'max_completion_tokens': self.max_completion_tokens,
            'messages': [m.to_json() for m in self.messages]
        }
    
    @staticmethod
    def from_dict(data):
        max_completion_tokens = data['max_completion_tokens'] if 'max_completion_tokens' in data else 1000
        stream = data['stream'] if 'stream' in data else False
        return ChatCompletionRequest(data['model'], stream, max_completion_tokens, [ChatMessage.from_dict(m) for m in data['messages']])

def send_initial_chunk(
    job: Job,
    created: int,
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
    created: int,
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
        complete_cb(req.model, req.messages, req.max_completion_tokens, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)
