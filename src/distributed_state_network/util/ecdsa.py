import hashlib
from typing import Tuple

from ecdsa import SigningKey, VerifyingKey, SECP256k1

def generate_key_pair() -> Tuple[bytes, bytes]:
    private_key = SigningKey.generate(curve=SECP256k1)
    public_key = private_key.get_verifying_key()
    if public_key is None:
        raise Exception("Error generating keys")
    return public_key.to_string(), private_key.to_string()

def sign_message(private_key_bytes: bytes, message: bytes) -> bytes:
    private_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    message_hash = hashlib.sha256(message).digest()
    return private_key.sign(message_hash)

def verify_signature(public_key: bytes, message: bytes, signature: bytes):
    public_key_obj = VerifyingKey.from_string(public_key, curve=SECP256k1)
    message_hash = hashlib.sha256(message).digest()
    try:
        return public_key_obj.verify(signature, message_hash)
    except Exception:
        return False
