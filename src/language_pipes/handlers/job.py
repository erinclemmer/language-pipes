import ssl
import threading
from threading import Thread
from typing import Tuple, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from distributed_state_network import DSNode

from language_pipes.util.http import _respond_bytes

class JobHandler(BaseHTTPRequestHandler):
    server: "JobServer"

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        Thread(target=self.server.cb, args=(body, )).start()
        _respond_bytes(self, b'UP' if not self.server.router.shutting_down else b'DOWN')

    def log_message(self, format, *args):
        pass

class JobServer(HTTPServer):
    router: DSNode
    cb: Callable # Callback into job receiver

    def __init__(
        self, 
        port: int,
        router: DSNode,
        cb: Callable
    ):
        super().__init__(("0.0.0.0", port), JobHandler)
        self.router = router
        self.cb = cb

    def stop(self):
        self.shutdown()
        self.socket.close()

    @staticmethod 
    def start(https: bool, cert_path: str, port: int, router: DSNode, cb: Callable) -> Tuple[Thread, 'JobServer']:
        httpd = JobServer(port, router, cb)
        if (https):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(
                certfile=cert_path,
                keyfile=cert_path.replace(".crt", ".key")
            )
            httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
        httpd_thread = threading.Thread(target=httpd.serve_forever, args=())
        httpd_thread.start()

        return httpd_thread, httpd
