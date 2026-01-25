import time
import random
import logging
import threading
import requests
from typing import Dict, List, Optional, Callable

from distributed_state_network.objects.endpoint import Endpoint
from distributed_state_network.objects.hello_packet import HelloPacket
from distributed_state_network.objects.peers_packet import PeersPacket
from distributed_state_network.objects.state_packet import StatePacket
from distributed_state_network.objects.data_packet import DataPacket
from distributed_state_network.objects.config import DSNodeConfig

from distributed_state_network.util import get_dict_hash
from distributed_state_network.util.key_manager import CredentialManager
from distributed_state_network.util.aes import aes_encrypt, aes_decrypt

TICK_INTERVAL = 3
HTTP_TIMEOUT = 2  # seconds

# Message type constants (must match handler.py)
MSG_HELLO = 1
MSG_PEERS = 2
MSG_UPDATE = 3
MSG_PING = 4
MSG_DATA = 5

# Map message types to endpoint paths
MSG_TYPE_TO_PATH = {
    MSG_HELLO: '/hello',
    MSG_PEERS: '/peers',
    MSG_UPDATE: '/update',
    MSG_PING: '/ping',
    MSG_DATA: '/data'
}

class DSNode:
    version: str
    config: DSNodeConfig
    address_book: Dict[str, Endpoint]
    node_states: Dict[str, StatePacket]
    shutting_down: bool

    def __init__(
            self, 
            config: DSNodeConfig,
            version: str,
            disconnect_callback: Optional[Callable] = None,
            update_callback: Optional[Callable] = None,
            receive_callback: Optional[Callable] = None
        ):
        self.config = config
        self.version = version
        self.shutting_down = False
        
        self.cred_manager = CredentialManager(config.credential_dir, config.node_id)
        self.cred_manager.generate_keys()
        
        self.node_states = {
            self.config.node_id: StatePacket.create(self.config.node_id, time.time(), self.cred_manager.my_private(), { })
        }

        self.address_book = {
            self.config.node_id: Endpoint(self.config.network_ip, config.port)
        }
        
        self.logger = logging.getLogger("DSN: " + config.node_id)
        self.disconnect_cb = disconnect_callback
        self.update_cb = update_callback
        self.receive_cb = receive_callback
        
        threading.Thread(target=self.network_tick, daemon=True).start()

    def get_aes_key(self) -> Optional[bytes]:
        if self.config.aes_key is None:
            return None
        return bytes.fromhex(self.config.aes_key)

    def write_address_book(self, node_id: str, conn: Endpoint):
        self.logger.info(f"Address set: {node_id} -> {conn.address}:{conn.port}")
        self.address_book[node_id] = conn

    def network_tick(self):
        time.sleep(TICK_INTERVAL)
        if self.shutting_down:
            self.logger.info("Shutting down node")
            return
        self.test_connections()
        self.gossip()
        threading.Thread(target=self.network_tick, daemon=True).start()

    def gossip(self):
        if len(self.address_book.keys()) == 0:
            return
        node_id = random.choice(list(self.address_book.keys()))
        if node_id == self.config.node_id:
            return
        
        self.send_update(node_id)

    def test_connections(self):
        def remove(node_id: str):
            if node_id in self.node_states:
                del self.node_states[node_id]
            if node_id in self.address_book:    
                del self.address_book[node_id]
            self.logger.info(f"PING failed for {node_id}, disconnecting...")
        
        for node_id in self.node_states.copy().keys():
            if node_id not in self.node_states or node_id == self.config.node_id:
                continue
            try:
                if self.shutting_down:
                    return
                self.send_ping(node_id)
            except Exception:
                self.logger.error(f"PING Failure for {node_id}")
                if node_id in self.node_states:  # double check if something has changed since the ping request started
                    remove(node_id)
                    if self.disconnect_cb is not None:
                        self.disconnect_cb()

    def send_http_request(self, endpoint: Endpoint, msg_type: int, payload: bytes, retries: int = 0) -> bytes:
        """Send HTTP request and wait for response"""
        try:
            # Prepend message type to payload
            data = bytes([msg_type]) + payload
            if self.config.aes_key is not None:
                data = self.encrypt_data(data)
            
            # Determine the URL path based on message type
            path = MSG_TYPE_TO_PATH.get(msg_type, '/unknown')
            url = f"http://{endpoint.address}:{endpoint.port}{path}"
            
            # Send HTTP POST request
            response = requests.post(
                url,
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=HTTP_TIMEOUT
            )
            
            # Check response status
            if response.status_code == 204:
                # No content - valid for some responses like successful HELLO with no data
                return b''
            elif response.status_code != 200:
                raise Exception(f"HTTP error: {response.status_code}")
            
            response_data = response.content
            # Decrypt the response
            if self.config.aes_key is not None:
                response_data = self.decrypt_data(response_data)
            
            if len(response_data) < 1:
                raise Exception("Empty response")
            
            # First byte is message type
            response_msg_type = response_data[0]
            if response_msg_type != msg_type:
                raise Exception(f"Response message type mismatch: expected {msg_type}, got {response_msg_type}")
            
            # Return the body (everything after the message type byte)
            return response_data[1:]
            
        except requests.exceptions.Timeout:
            if retries < 2:
                time.sleep(0.5)
                return self.send_http_request(endpoint, msg_type, payload, retries + 1)
            else:
                raise Exception(f"HTTP request to {endpoint.to_string()} timed out")
        except requests.exceptions.RequestException as e:
            if retries < 2:
                time.sleep(0.5)
                return self.send_http_request(endpoint, msg_type, payload, retries + 1)
            else:
                raise Exception(f"HTTP request to {endpoint.to_string()} failed: {e}")
        except Exception as e:
            if retries < 2:
                time.sleep(0.5)
                return self.send_http_request(endpoint, msg_type, payload, retries + 1)
            else:
                raise Exception(f"HTTP request to {endpoint.to_string()} failed: {e}")

    def send_request_to_node(self, node_id: str, msg_type: int, payload: bytes) -> bytes:
        con = self.connection_from_node(node_id)
        return self.send_http_request(con, msg_type, payload)

    def encrypt_data(self, data: bytes) -> bytes:
        key = self.get_aes_key()
        if key is None:
            return data
        return aes_encrypt(key, data)

    def decrypt_data(self, data: bytes) -> bytes:
        key = self.get_aes_key()
        if key is None:
            return data
        return aes_decrypt(key, data)

    def request_peers(self, node_id: str):
        pkt = PeersPacket(self.config.node_id, None, { })
        pkt.sign(self.cred_manager.my_private())
        content = self.send_request_to_node(node_id, MSG_PEERS, pkt.to_bytes())
        pkt = PeersPacket.from_bytes(content)
        if not pkt.verify_signature(self.cred_manager.read_public(node_id)):
            raise Exception("Could not verify peers packet")

        for key in pkt.connections.keys():
            if key == self.config.node_id:
                continue
            
            self.write_address_book(key, pkt.connections[key])
            
            if key not in self.node_states:
                self.send_hello(self.address_book[key])
                
            node_state = self.send_update(key)
            self.handle_update(node_state)

    def handle_peers(self, data: bytes):
        pkt = PeersPacket.from_bytes(data)
        if pkt.node_id not in self.address_book:
            raise Exception(401, f"Could not find {pkt.node_id} in address book")  # Not Authorized
        
        if not pkt.verify_signature(self.cred_manager.read_public(pkt.node_id)):
            raise Exception(406, "Could not verify ECDSA signature of packet")  # Not Acceptable

        peers = { }
        for key in self.address_book.keys():
            if key == self.config.node_id:
                continue
            peers[key] = self.address_book[key]
        
        pkt = PeersPacket(self.config.node_id, None, peers)
        pkt.sign(self.cred_manager.my_private())
        return pkt.to_bytes()

    def send_hello(self, con: Endpoint):
        self.logger.info(f"HELLO => {con.to_string()}")

        pkt = self.my_hello_packet()
        payload = pkt.to_bytes()
        content = self.send_http_request(con, MSG_HELLO, payload)
        
        # Get the response packet
        pkt = HelloPacket.from_bytes(content)
        self.logger.info(f"Received HELLO from {pkt.node_id}")
        
        # Verify version compatibility
        if pkt.version != self.version:
            msg = f"HELLO => {pkt.node_id} (Version mismatch \"{pkt.version}\" != \"{self.version}\")"
            self.logger.error(msg)
            raise Exception(505)  # Version not supported

        # Store the peer's public key
        self.cred_manager.ensure_public(pkt.node_id, pkt.ecdsa_public_key)
        
        # If the server sent us our detected IP, update our address book
        if pkt.detected_address:
            self.logger.info(f"Server detected our IP as: {pkt.detected_address}")
            # Update our own connection in the address book with the detected IP
            self.write_address_book(self.config.node_id, Endpoint(pkt.detected_address, self.config.port))
        
        self.write_address_book(pkt.node_id, con)

        if pkt.node_id not in self.node_states:
            self.init_state(pkt.node_id)

        return pkt.node_id

    def init_state(self, node_id: str):
        self.node_states[node_id] = StatePacket(node_id, 0, b'', { })

    def handle_hello(self, data: bytes, detected_address: str) -> bytes:
        pkt = HelloPacket.from_bytes(data)
        self.logger.info(f"Received HELLO from {pkt.node_id}")
        if pkt.version != self.version:
            msg = f"HELLO => {pkt.node_id} (Version mismatch \"{pkt.version}\" != \"{self.version}\")"
            self.logger.error(msg)
            raise Exception(505)  # Version not supported

        self.cred_manager.ensure_public(pkt.node_id, pkt.ecdsa_public_key)
        self.write_address_book(pkt.node_id, Endpoint(detected_address, pkt.connection.port))

        if pkt.node_id not in self.node_states:
            self.init_state(pkt.node_id)

        # Create response with detected address
        response_pkt = self.my_hello_packet()
        response_pkt.detected_address = detected_address
        return response_pkt.to_bytes()

    def my_hello_packet(self) -> HelloPacket:
        pkt = HelloPacket(
            self.version, 
            self.config.node_id, 
            self.my_con(), 
            self.cred_manager.my_public(), 
            None,
            None  # No certificate for HTTP
        )
        pkt.sign(self.cred_manager.my_private())
        return pkt

    def send_ping(self, node_id: str):     
        try:
            conn = self.connection_from_node(node_id)
            self.logger.debug(f"PING => {node_id} ({conn.address}:{conn.port})")
            self.send_request_to_node(node_id, MSG_PING, b' ')
        except Exception as e:
            raise Exception(f'PING => {node_id}: {e}')

    def send_update(self, node_id: str):
        self.logger.debug(f"UPDATE => {node_id}")
        content = self.send_request_to_node(node_id, MSG_UPDATE, self.my_state().to_bytes())
        return content

    def handle_update(self, data: bytes):
        pkt = StatePacket.from_bytes(data)
        
        if not self.update_state(pkt):
            return b''

        self.logger.info(f"Received UPDATE from {pkt.node_id}")

        if self.update_cb is not None:
            try:
                self.update_cb()
            except Exception as e:
                self.logger.error("Update Error Captured:")
                self.logger.error(str(e))

        return self.my_state().to_bytes()

    def my_state(self):
        return self.node_states[self.config.node_id]
    
    def update_state(self, pkt: StatePacket) -> bool:
        # ignore if we accidentally sent an update to ourselves
        if pkt.node_id == self.config.node_id:
            raise Exception(406, "Origin and destination are the same")  # Not acceptable

        if pkt.node_id in self.address_book and not pkt.verify_signature(self.cred_manager.read_public(pkt.node_id)):
            raise Exception(401, "Could not verify ECDSA signature")  # Not authorized

        if pkt.node_id in self.node_states:
            current_state = self.node_states[pkt.node_id]

            # Check if stale
            if current_state.last_update > pkt.last_update:
                return False

            # Check if duplicate packet
            if len(pkt.state_data.keys()) > 0 and get_dict_hash(self.node_states[pkt.node_id].state_data) == get_dict_hash(pkt.state_data):
                return False

        self.node_states[pkt.node_id] = pkt
        return True

    def bootstrap(self, con: Endpoint):
        bootstrap_id = self.send_hello(con)
        content = self.send_update(bootstrap_id)
        self.handle_update(content)
        self.request_peers(bootstrap_id)

    def connection_from_node(self, node_id: str) -> Endpoint:
        if node_id not in self.address_book:
            raise Exception(f"could not find connection for {node_id}")
        return self.address_book[node_id]

    def update_data(self, key: str, val: str):
        self.node_states[self.config.node_id].update_state(key, val, self.cred_manager.my_private())
        for key in list(self.node_states.keys())[:]:
            if key == self.config.node_id:
                continue
            try:
                self.send_update(key)
            except Exception as e:
                print(e)

    def my_con(self) -> Endpoint:
        return self.connection_from_node(self.config.node_id)

    def read_data(self, node_id: str, key: str) -> Optional[str]:
        if key not in self.node_states[node_id].state_data.keys():
            return None
        return self.node_states[node_id].state_data[key]

    def peers(self) -> List[str]:
        return list(self.node_states.keys())

    def send_to_node(self, node_id: str, data: bytes) -> str:
        pkt = DataPacket.create(self.config.node_id, self.cred_manager.my_private(), data)
        response = self.send_request_to_node(node_id, MSG_DATA, pkt.to_bytes())
        try:
            return response.decode('utf-8')
        except Exception:
            return ''
    
    def receive_data(self, data: bytes):
        pkt = DataPacket.from_bytes(data)
        self.logger.info(f"Received DATA from {pkt.node_id} ({len(pkt.data)} bytes)")
        
        # Ensure sender is known
        if pkt.node_id not in self.address_book:
            raise Exception(401, f"Could not find {pkt.node_id} in address book")
        
        # Verify signature using stored public key
        if not pkt.verify_signature(self.cred_manager.read_public(pkt.node_id)):
            raise Exception(401, "Could not verify ECDSA signature of data packet")
        
        if self.receive_cb is not None:
            try:
                self.receive_cb(pkt.data)
            except Exception as e:
                print(e)

        return b'OK'
