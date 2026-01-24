from distributed_state_network.util.ecdsa import verify_signature, sign_message

class SignedPacket:
    ecdsa_signature: bytes

    def __init__(self, ecdsa_signature: bytes):
        self.ecdsa_signature = ecdsa_signature

    def sign(self, private_key: bytes):
        self.ecdsa_signature = sign_message(private_key, self.to_bytes(False))

    def verify_signature(self, public_key: bytes):
        return verify_signature(public_key, self.to_bytes(False), self.ecdsa_signature)

    def to_bytes(include_signature: bool = True):
        pass

    @staticmethod
    def from_bytes():
        pass