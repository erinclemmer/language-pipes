import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional

from language_pipes.util.oai import oai_chat_complete, get_models
from language_pipes.util.http import _send_code

class T:
    complete: Callable
    api_keys: List[str]

class OAIHttpHandler(BaseHTTPRequestHandler):
    server: T

    def _validate_key(self, key: str) -> bool:
        if len(self.server.api_keys) == 0:
            return True
        return key in self.server.api_keys
    
    def _extract_api_key(self, header: str) -> Optional[str]:
        if not "Bearer " in header:
            return None
        return header[7:]
    
    def authorize(self) -> bool:
        api_key_header = self.headers.get("Authorization", None) 
        if api_key_header is None:
            _send_code(400, self, "No authorization token supplied")
            return False
        api_key = self._extract_api_key(api_key_header)
        if api_key is None or api_key not in self.server.api_keys:
            _send_code(401, self, "Unauthorized")
            return False
        return True

    def do_POST(self):
        if len(self.server.api_keys) > 0 and not self.authorize():
            return
                
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
    
    def __init__(self, port: int, api_keys: List[str], complete: Callable, get_models: Callable):
        super().__init__(("0.0.0.0", port), OAIHttpHandler)
        self.api_keys = api_keys
        self.complete = complete
        self.get_models = get_models
