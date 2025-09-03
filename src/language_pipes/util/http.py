import json
from http.server import BaseHTTPRequestHandler

def _respond_bytes(handler: BaseHTTPRequestHandler, data: bytes):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.end_headers()
    handler.wfile.write(data)
    handler.wfile.flush()

def _respond_json(handler: BaseHTTPRequestHandler, data):
    response = json.dumps(data).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(response)
    handler.wfile.flush()

def _send_code(code: int, handler: BaseHTTPRequestHandler, message: str):
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(message.encode())
    handler.wfile.flush()
