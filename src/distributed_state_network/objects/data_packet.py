from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

class DataPacket(SignedPacket):
    node_id: str
    data: bytes

    def __init__(self, node_id: str, ecdsa_signature: bytes, data: bytes):
        super().__init__(ecdsa_signature)
        self.node_id = node_id
        self.data = data

    def to_bytes(self, include_signature: bool = True) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.node_id)
        if include_signature:
            bts.write_bytes(self.ecdsa_signature)
        bts.write_bytes(self.data)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> "DataPacket":
        bts = ByteHelper(data)
        node_id = bts.read_string()
        if node_id == "":
            raise Exception(406, "Malformed packet")
        ecdsa_signature = bts.read_bytes()
        payload = bts.read_bytes()
        return DataPacket(node_id, ecdsa_signature, payload)

    @staticmethod
    def create(node_id: str, ecdsa_private_key: bytes, data: bytes) -> "DataPacket":
        pkt = DataPacket(node_id, b"", data)
        pkt.sign(ecdsa_private_key)
        return pkt
