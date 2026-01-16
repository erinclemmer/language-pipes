import json
from http.server import BaseHTTPRequestHandler

def _respond_json(handler: BaseHTTPRequestHandler, data):
    response = json.dumps(data).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(response)
    handler.wfile.flush()

def _send_sse_headers(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

def _send_code(code: int, handler: BaseHTTPRequestHandler, message: str):
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(message.encode())
    handler.wfile.flush()
