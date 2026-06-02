
from http.server import BaseHTTPRequestHandler
from typing import Optional

from flask import json

from language_pipes.jobs.job import Job


def send_initial_chunk(
    job: Job,
    created: float,
    handler: BaseHTTPRequestHandler
):
    msg = {
        "id": f"chatcmpl-{job.job_id}",
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": job.model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }
        ]
    }
    data_bytes = json.dumps(msg).encode('utf-8')
    try:
        handler.wfile.write(b'data: ' + data_bytes + b'\n\n')
        handler.wfile.flush()
    except Exception:
        pass

def send_update_chunk(
    job: Job,
    delta: object,
    created: float,
    finish_reason: Optional[str],
    handler: BaseHTTPRequestHandler
):
    msg = {
        "id": f"chatcmpl-{job.job_id}",
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
    except Exception:
        return False
    return True

def send_complete(job: Job, created: float, handler: BaseHTTPRequestHandler):
    final = {
        "id": f"chatcmpl-{job.job_id}",
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": job.model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    try:
        handler.wfile.write(b'data: ' + json.dumps(final).encode('utf-8') + b'\n\n')
        handler.wfile.write(b'data: [DONE]\n\n')
        handler.wfile.flush()
    except Exception:
        pass