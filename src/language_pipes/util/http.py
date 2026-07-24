import json
import select
import socket
from http.server import BaseHTTPRequestHandler

def _connection_alive(handler: BaseHTTPRequestHandler) -> bool:
    """Check whether the client socket is still connected without writing to it.

    Used where a request has nothing to stream yet (non-streaming responses,
    or buffered tool-call output) so a dropped connection would otherwise go
    undetected until generation finishes on its own.
    """
    sock = getattr(handler, "connection", None)
    if sock is None:
        return True
    try:
        readable, _, errored = select.select([sock], [], [sock], 0)
        if errored:
            return False
        return not (readable and sock.recv(1, socket.MSG_PEEK) == b"")
    except OSError:
        return False

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
    handler.send_header("Connection", "close")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

def _send_code(code: int, handler: BaseHTTPRequestHandler, message: str):
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(message.encode())
    handler.wfile.flush()
