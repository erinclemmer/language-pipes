from typing import Optional

from distributed_state_network.objects.endpoint import Endpoint

from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

class HelloPacket(SignedPacket):
    version: str
    node_id: str
    connection: Endpoint
    ecdsa_public_key: bytes
    detected_address: Optional[str]  # IP address detected by bootstrap node

    def __init__(
        self, 
        version: str, 
        node_id: str, 
        connection: Endpoint,
        ecdsa_public_key: bytes,
        ecdsa_signature: bytes,
        detected_address: Optional[str] = None
    ):
        super().__init__(ecdsa_signature)
        self.version = version
        self.node_id = node_id
        self.connection = connection
        self.ecdsa_public_key = ecdsa_public_key
        self.detected_address = detected_address

    def to_bytes(self, include_signature: bool = True):
        bts = ByteHelper()
        bts.write_string(self.version)
        bts.write_string(self.node_id)
        if self.connection.address is not None:
            bts.write_bytes(self.connection.address.encode())
        else:
            bts.write_bytes(b'')
        bts.write_int(self.connection.port)
        bts.write_bytes(self.ecdsa_public_key)
        if include_signature:
            bts.write_bytes(self.ecdsa_signature)
        # Add detected address (empty string if None)
        bts.write_string(self.detected_address or "")
        
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        version = bts.read_string()
        node_id = bts.read_string()
        addr_bytes = bts.read_bytes()
        address = None
        if addr_bytes != b'':
            address = addr_bytes.decode()
        port = bts.read_int()

        connection = Endpoint(
            address=address,
            port=port
        )
        ecdsa_public_key = bts.read_bytes()
        ecdsa_signature = bts.read_bytes()
        # Read detected address (may be empty string for older packets)
        detected_address = bts.read_string() or None

        if version == '' or node_id == '' or ecdsa_public_key == b'':
            raise Exception(406, "Malformed packet") # Not acceptable

        return HelloPacket(version, node_id, connection, ecdsa_public_key, ecdsa_signature, detected_address)
