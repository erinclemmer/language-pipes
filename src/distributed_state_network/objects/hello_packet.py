from typing import Optional

from distributed_state_network.objects.endpoint import Endpoint

from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

class HelloPacket(SignedPacket):
    version: str
    node_id: str
    connection: Endpoint
    ecdsa_public_key: bytes
    https_certificate: bytes
    detected_address: Optional[str]  # IP address detected by the server

    def __init__(
        self, 
        version: str, 
        node_id: str, 
        connection: Endpoint,
        ecdsa_public_key: bytes,
        ecdsa_signature: bytes,
        https_certificate: bytes,
        detected_address: Optional[str] = None
    ):
        super().__init__(ecdsa_signature)
        self.version = version
        self.node_id = node_id
        self.connection = connection
        self.ecdsa_public_key = ecdsa_public_key
        self.https_certificate = https_certificate
        self.detected_address = detected_address

    def to_bytes(self, include_signature: bool = True):
        bts = ByteHelper()
        bts.write_string(self.version)
        bts.write_string(self.node_id)
        bts.write_bytes(self.connection.to_bytes())
        bts.write_bytes(self.ecdsa_public_key)
        if include_signature:
            bts.write_bytes(self.ecdsa_signature)
        if self.https_certificate is not None:
            bts.write_bytes(self.https_certificate)
        # Add detected address (empty string if None)
        bts.write_string(self.detected_address or "")
        
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        version = bts.read_string()
        node_id = bts.read_string()
        connection = Endpoint.from_bytes(bts.read_bytes())
        ecdsa_public_key = bts.read_bytes()
        ecdsa_signature = bts.read_bytes()
        https_certificate = bts.read_bytes() or None
        # Read detected address (may be empty string for older packets)
        detected_address = bts.read_string() or None

        if version == '' or node_id == '' or ecdsa_public_key == b'':
            raise Exception(406, "Malformed packet") # Not acceptable

        return HelloPacket(version, node_id, connection, ecdsa_public_key, ecdsa_signature, https_certificate, detected_address)
