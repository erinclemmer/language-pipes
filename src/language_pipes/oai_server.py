
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from language_pipes.util.oai import oai_chat_complete, get_models
from language_pipes.util.http import _send_code

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
            _send_code(400, self, "Invalid JSON")
            return
        
        if 'model' not in data:
            _send_code(400, self, "model parameter is required")
            return

        if 'messages' not in data:
            _send_code(400, self, "messages object parameter is required")
            return

        if len(data['messages']) == 0:
            _send_code(400, self, "messages object must not be empty")
            return
        
        if self.path == '/v1/chat/completions':
            oai_chat_complete(self, self.server.complete, data)

    def do_GET(self):
        if self.path == '/v1/models':
            get_models(self, self.server.get_models)

class OAIHttpServer(ThreadingHTTPServer):
    complete: Callable
    
    def __init__(self, port: int, complete: Callable, get_models: Callable):
        super().__init__(("0.0.0.0", port), OAIHttpHandler)
        self.complete = complete
        self.get_models = get_models
