import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import replace
from typing import Callable, List, Optional, Tuple, cast
from distributed_state_network.dsnode import DSNode
from distributed_state_network.objects.config import DSNodeConfig
from distributed_state_network.util.aes import generate_aes_key
from distributed_state_network.util import stop_thread
from distributed_state_network.network_protocol import StateNetworkNode

VERSION = "0.9.0"

# Message type constants
MSG_HELLO = 1
MSG_PEERS = 2
MSG_UPDATE = 3
MSG_PING = 4
MSG_DATA = 5

PATH_TO_MSG_TYPE = {
    '/hello': MSG_HELLO,
    '/peers': MSG_PEERS,
    '/update': MSG_UPDATE,
    '/ping': MSG_PING,
    '/data': MSG_DATA,
}


class _DSNodeHTTPServer(ThreadingHTTPServer):
    dsnode_server: 'DSNodeServer'

    def __init__(self, server_address: Tuple[str, int], dsnode_server: 'DSNodeServer'):
        super().__init__(server_address, _DSNodeHTTPRequestHandler)
        self.dsnode_server = dsnode_server


class _DSNodeHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        msg_type = PATH_TO_MSG_TYPE.get(self.path)
        if msg_type is None:
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(content_length)
        server = cast(_DSNodeHTTPServer, self.server)
        status, response_data = server.dsnode_server._handle_request(
            msg_type,
            data,
            self.client_address[0] if self.client_address else None,
        )

        self.send_response(status)
        if response_data is not None:
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()

        if response_data is not None:
            try:
                self.wfile.write(response_data)
            except BrokenPipeError:
                pass

    def log_message(self, format: str, *args):
        return

class DSNodeServer(StateNetworkNode):
    config: DSNodeConfig
    network_ip: Optional[str]
    running: bool
    node: DSNode
    thread: Optional[threading.Thread]
    http_server: Optional[_DSNodeHTTPServer]
    create_alert: Callable[[str], None]

    def __init__(
        self, 
        config: DSNodeConfig,
        create_alert: Callable[[str], None],
        disconnect_callback: Optional[Callable] = None,
        update_callback: Optional[Callable] = None,
        receive_callback: Optional[Callable] = None,
    ):
        detected_ip = self._detect_local_ip() if config.network_ip is None else config.network_ip
        self.network_ip = detected_ip
        self.config = replace(config, network_ip=detected_ip) if config.network_ip != detected_ip else config
        self.running = False
        self.thread = None
        self.http_server = None
        self.create_alert = create_alert
        
        # Create DSNode
        self.node = DSNode(self.config, VERSION, create_alert, disconnect_callback, update_callback, receive_callback)

    def _detect_local_ip(self) -> Optional[str]:
        """Best-effort local network IP detection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # No traffic is sent, but this lets the OS select the outbound interface.
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass

        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass

        return None

    def _handle_request(self, msg_type: int, data: bytes, remote_addr: Optional[str]) -> Tuple[int, Optional[bytes]]:
        if not self.running:
            return 500, None
        try:
            # Decrypt the data
            if self.config.aes_key is not None:
                try:
                    data = self.node.decrypt_data(data)
                except Exception:
                    return 401, b"Missing or incorrect encryption key"
            
            if len(data) < 1:
                return 400, None
            
            # First byte should be message type (for verification)
            received_msg_type = data[0]
            body = data[1:]
            
            if received_msg_type != msg_type:
                self.node.logger.error(f"Message type mismatch: expected {msg_type}, got {received_msg_type}")
                return 400, None
            
            response_data = None
            
            if msg_type == MSG_HELLO:
                # Pass the detected IP address to handle_hello
                if remote_addr is None:
                    raise ValueError("Must supply remote address with hello")
                response_data = self.node.handle_hello(body, remote_addr)
                
            elif msg_type == MSG_PEERS:
                response_data = self.node.handle_peers(body)
                
            elif msg_type == MSG_UPDATE:
                response_data = self.node.handle_update(body)
                
            elif msg_type == MSG_PING:
                response_data = b''

            elif msg_type == MSG_DATA:
                response_data = self.node.receive_data(body)
            
            # Send response if handler returned data
            if response_data is not None:
                # Prepend message type to response
                response_with_type = bytes([msg_type]) + response_data
                if self.config.aes_key is not None:
                    response_with_type = self.node.encrypt_data(response_with_type)
                return 200, response_with_type
            else:
                return 204, None  # No content
                
        except Exception as e:
            if len(e.args) >= 2 and isinstance(e.args[0], int):
                # Error with HTTP status code
                self.node.logger.error(f"Error handling {msg_type} from {remote_addr}: {e.args[1]}")
                return e.args[0], e.args[1].encode()
            else:
                self.node.logger.error(f"Error handling {msg_type} from {remote_addr}: {e}")
                return 500, None

    def stop(self):
        self.node.shutting_down = True
        self.running = False
        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None
        if self.thread is not None:
            stop_thread(self.thread)

    def _serve_forever(self, port: int):
        if self.running:
            return

        self.running = True
        try:
            self.http_server = _DSNodeHTTPServer(('0.0.0.0', port), self)
            self.node.logger.info(f'Started DSNode on HTTP port {port}')
            self.http_server.serve_forever()
        except Exception as e:
            self.node.add_log(str(e), "ERROR")
            return

    @staticmethod
    def generate_key() -> str:
        return generate_aes_key().hex()


    @staticmethod 
    def start(
        config: DSNodeConfig, 
        create_alert: Callable[[str], None],
        disconnect_callback: Optional[Callable] = None, 
        update_callback: Optional[Callable] = None,
        receive_callback: Optional[Callable] = None
    ) -> 'DSNodeServer':
        n = DSNodeServer(config, create_alert, disconnect_callback, update_callback, receive_callback)
        n.thread = threading.Thread(target=n._serve_forever, daemon=True, args=(config.port, ))
        n.thread.start()

        if n.config.bootstrap_nodes is not None and len(n.config.bootstrap_nodes) > 0:
            connected = False
            for bs in n.config.bootstrap_nodes:
                try:
                    n.node.bootstrap(bs)
                    connected = True
                    break # Throws exception if connection is not made
                except Exception as e:
                    n.node.logger.error(e)

            if not connected:
                n.create_alert("Could not connect to any bootstrap node")

        return n

    def peers(self) -> List[str]:
        return self.node.peers()
    
    def read_data(self, node_id: str, key: str) -> Optional[str]:
        return self.node.read_data(node_id, key)
    
    def update_data(self, key: str, value: str):
        self.node.update_data(key, value)

    def send_to_node(self, node_id: str, data: bytes):
        self.node.send_to_node(node_id, data)

    def is_shut_down(self) -> bool:
        return self.node.shutting_down
    
    def node_id(self) -> str:
        return self.config.node_id

    def set_receive_cb(self, cb: Callable):
        self.node.receive_cb = cb

    def set_update_cb(self, cb: Callable):
        self.node.update_cb = cb

    def set_disconnect_cb(self, cb: Callable):
        self.node.disconnect_cb = cb

    def receive_data(self, data: bytes):
        if self.node.receive_cb is not None:
            self.node.receive_cb(self.config.node_id, data)
