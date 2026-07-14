from typing import Optional
from distributed_state_network.util.ecdsa import verify_signature, sign_message

class SignedPacket:
    ecdsa_signature: Optional[bytes]

    def __init__(self, ecdsa_signature: Optional[bytes]):
        self.ecdsa_signature = ecdsa_signature

    def sign(self, private_key: bytes):
        self.ecdsa_signature = sign_message(private_key, self.to_bytes(False))

    def verify_signature(self, public_key: bytes):
        if self.ecdsa_signature is None:
            raise ValueError("Cannot verifiy, ECDSA signature was not set")
        return verify_signature(public_key, self.to_bytes(False), self.ecdsa_signature)

    def to_bytes(self, include_signature: bool = True) -> bytes:
        return b''

    @staticmethod
    def from_bytes(data: bytes) -> "SignedPacket":
        return SignedPacket(b'')
