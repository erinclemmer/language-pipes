import time
import json
from typing import Dict

from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

class StatePacket(SignedPacket):
    node_id: str
    state_data: Dict[str, str]
    last_update: float

    def __init__(
            self, 
            node_id: str,
            last_update: float,
            ecdsa_signature: bytes,
            state_data: Dict[str, str]
        ):
        super().__init__(ecdsa_signature)
        self.node_id = node_id
        self.state_data = state_data
        self.last_update = last_update

    def update_state(self, key: str, val: str, private_key: bytes):
        self.state_data[key] = val
        self.last_update = time.time()
        self.sign(private_key)

    def to_bytes(self, include_signature: bool = True):
        bts = ByteHelper()
        bts.write_string(self.node_id)
        bts.write_float(self.last_update)
        if include_signature:
            bts.write_bytes(self.ecdsa_signature)
        bts.write_string(json.dumps(self.state_data))

        return bts.get_bytes()
    
    @staticmethod
    def create(
        node_id: str, 
        last_update: float,
        ecdsa_private_key: bytes,
        state_data: Dict[str, str]
    ):
        s = StatePacket(node_id, last_update, b'', state_data)
        s.sign(ecdsa_private_key)
        return s


    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        node_id = bts.read_string()
        
        if node_id == '':
            raise Exception(406, "Malformed packet")
        
        last_update = bts.read_float()
        ecdsa_signature = bts.read_bytes()
        state_data = json.loads(bts.read_string())

        return StatePacket(node_id, last_update, ecdsa_signature, state_data)
