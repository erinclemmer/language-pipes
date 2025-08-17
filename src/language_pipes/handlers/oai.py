
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

from language_pipes.util.oai import oai_chat_complete
from language_pipes.util.http import _send_403

class T:
    complete: Callable

class OAIHttpHandler(BaseHTTPRequestHandler):
    server: T

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            _send_403(self, "Invalid JSON")
            return
        
        if self.path == '/v1/chat/completions':
            oai_chat_complete(self, self.server.complete, data)

class OAIHttpServer(HTTPServer):
    complete: Callable
    
    def __init__(self, port: int, complete: Callable):
        super().__init__(("0.0.0.0", port), OAIHttpHandler)
        self.complete = complete
