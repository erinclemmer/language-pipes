import time
from typing import Callable, List

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.util.chat import ChatMessage
from language_pipes.util.http import _respond_json
from language_pipes.job_manager.job import Job

class ChatCompletionRequest:
    model: str
    messages: List[ChatMessage]
    max_completion_tokens: int

    def __init__(
            self, 
            model: str, 
            max_completion_tokens: int,
            messages: List[ChatMessage]
        ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.messages = messages

    def to_json(self):
        return {
            'model': self.model,
            'max_completion_tokens': self.max_completion_tokens,
            'messages': [m.to_json() for m in self.messages]
        }
    
    @staticmethod
    def from_dict(data):
        max_completion_tokens = data['max_completion_tokens'] if 'max_completion_tokens' in data else 1000
        return ChatCompletionRequest(data['model'], max_completion_tokens, [ChatMessage.from_dict(m) for m in data['messages']])

def oai_chat_complete(handler: BaseHTTPRequestHandler, complete: Callable, data: dict):
    req = ChatCompletionRequest.from_dict(data)
    created_at = time.time()
    def cb(job: Job):
        if type(job) == type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available"})
        else:
            _respond_json(handler, {
                "id": job.job_id,
                "object": "chat.completion",
                "created": time.strftime("%m/%d/%Y:%H:%M:%S", time.localtime(created_at)),
                "model": "",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": job.result
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": job.prompt_tokens,
                    "completion_tokens": job.current_token,
                    "total_tokens": job.prompt_tokens + job.current_token
                }
            })

    def promise_fn(res: Callable, _: Callable):
        complete(req.model, req.messages, req.max_completion_tokens, res)
    job = Promise(promise_fn).get()
    cb(job)
