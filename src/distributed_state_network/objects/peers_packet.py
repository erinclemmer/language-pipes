from typing import Dict, Optional

from distributed_state_network.objects.endpoint import Endpoint
from distributed_state_network.util.byte_helper import ByteHelper
from distributed_state_network.objects.signed_packet import SignedPacket

class PeersPacket(SignedPacket):
    node_id: str
    connections: Dict[str, Endpoint]

    def __init__(self, node_id: str, ecdsa_signature: Optional[bytes], connections: Dict[str, Endpoint]):
        super().__init__(ecdsa_signature)
        self.node_id = node_id
        self.connections = connections

    def to_bytes(self, include_signature: bool = True) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.node_id)
        if include_signature :
            if self.ecdsa_signature is None:
                raise ValueError("Cannot convert unsigned packet to bytes")
            bts.write_bytes(self.ecdsa_signature)
        
        bts.write_int(len(self.connections.keys()))
        for key in self.connections.keys():
            bts.write_string(key)
            bts.write_bytes(self.connections[key].to_bytes())
        
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        node_id = bts.read_string()
        ecdsa_signature = bts.read_bytes()

        connections = { }
        num_keys = bts.read_int()
        for _ in range(num_keys):
            key = bts.read_string()
            connections[key] = Endpoint.from_bytes(bts.read_bytes())
        
        return PeersPacket(node_id, ecdsa_signature, connections)
