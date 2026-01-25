import sys
import threading
import logging
from typing import Callable, List, Optional
from flask import Flask, request, Response
from distributed_state_network.dsnode import DSNode
from distributed_state_network.objects.config import DSNodeConfig
from distributed_state_network.util.aes import generate_aes_key
from distributed_state_network.util import stop_thread

VERSION = "0.6.7"
logging.basicConfig(level=logging.INFO)

# Silence Flask and Werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Message type constants
MSG_HELLO = 1
MSG_PEERS = 2
MSG_UPDATE = 3
MSG_PING = 4
MSG_DATA = 5

class DSNodeServer:
    config: DSNodeConfig
    running: bool
    app: Flask
    node: DSNode
    thread: Optional[threading.Thread]

    def __init__(
        self, 
        config: DSNodeConfig,
        disconnect_callback: Optional[Callable] = None,
        update_callback: Optional[Callable] = None,
        receive_callback: Optional[Callable] = None
    ):
        self.config = config
        self.running = False
        self.thread = None
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)  # Reduce Flask's logging noise
        
        # Create DSNode
        self.node = DSNode(config, VERSION, disconnect_callback, update_callback, receive_callback)
        
        # Set up Flask routes
        self._setup_routes()

    def _setup_routes(self):
        """Set up Flask routes for each message type"""
        
        @self.app.route('/hello', methods=['POST'])
        def handle_hello_route():
            return self._handle_request(MSG_HELLO, request.data, request.remote_addr)
        
        @self.app.route('/peers', methods=['POST'])
        def handle_peers_route():
            return self._handle_request(MSG_PEERS, request.data, request.remote_addr)
        
        @self.app.route('/update', methods=['POST'])
        def handle_update_route():
            return self._handle_request(MSG_UPDATE, request.data, request.remote_addr)
        
        @self.app.route('/ping', methods=['POST'])
        def handle_ping_route():
            return self._handle_request(MSG_PING, request.data, request.remote_addr)

        @self.app.route('/data', methods=['POST'])
        def handle_data_route():
            return self._handle_request(MSG_DATA, request.data, request.remote_addr)


    def _handle_request(self, msg_type: int, data: bytes, remote_addr: Optional[str]) -> Response:
        if not self.running:
            return Response(status=500)
        try:
            # Decrypt the data
            if self.config.aes_key is not None:
                data = self.node.decrypt_data(data)
            
            if len(data) < 1:
                return Response(status=400)
            
            # First byte should be message type (for verification)
            received_msg_type = data[0]
            body = data[1:]
            
            if received_msg_type != msg_type:
                self.node.logger.error(f"Message type mismatch: expected {msg_type}, got {received_msg_type}")
                return Response(status=400)
            
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
                return Response(response_with_type, status=200, mimetype='application/octet-stream')
            else:
                return Response(status=204)  # No content
                
        except Exception as e:
            if len(e.args) >= 2 and isinstance(e.args[0], int):
                # Error with HTTP status code
                self.node.logger.error(f"Error handling {msg_type} from {remote_addr}: {e.args[1]}")
                return Response(status=e.args[0])
            else:
                self.node.logger.error(f"Error handling {msg_type} from {remote_addr}: {e}")
                return Response(status=500)

    def stop(self):
        self.node.shutting_down = True
        self.running = False
        if self.thread is not None:
            stop_thread(self.thread)

    def _serve_forever(self, port: int):
        if self.running:
            return
        # Suppress Flask startup messages
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        
        self.running = True
        self.app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
        self.node.logger.info(f'Started DSNode on HTTP port {port}')

    @staticmethod
    def generate_key() -> str:
        return generate_aes_key().hex()

    @staticmethod 
    def start(
        config: DSNodeConfig, 
        disconnect_callback: Optional[Callable] = None, 
        update_callback: Optional[Callable] = None,
        receive_callback: Optional[Callable] = None
    ) -> 'DSNodeServer':
        n = DSNodeServer(config, disconnect_callback, update_callback, receive_callback)
        n.thread = threading.Thread(target=n._serve_forever, daemon=True, args=(config.port, ))
        n.thread.start()

        if n.config.bootstrap_nodes is not None and len(n.config.bootstrap_nodes) > 0:
            for bs in n.config.bootstrap_nodes:
                try:
                    n.node.bootstrap(bs)
                    break # Throws exception if connection is not made
                except Exception as e:
                    print(e)

        return n

    def peers(self) -> List[str]:
        return self.node.peers()
    
    def read_data(self, node_id: str, key: str) -> Optional[str]:
        return self.node.read_data(node_id, key)
    
    def update_data(self, key: str, val: str):
        return self.node.update_data(key, val)

    def send_to_node(self, node_id: str, data: bytes):
        return self.node.send_to_node(node_id, data)

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

    def _receive_data(self, data: bytes):
        if self.node.receive_cb is not None:
            self.node.receive_cb(data)
