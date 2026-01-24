import hashlib

from ecdsa import SigningKey, VerifyingKey, SECP256k1

def generate_key_pair(_):
    private_key = SigningKey.generate(curve=SECP256k1)
    public_key = private_key.get_verifying_key()
    return public_key.to_string(), private_key.to_string()

def sign_message(private_key: bytes, message: bytes) -> bytes:
    private_key = SigningKey.from_string(private_key, curve=SECP256k1)
    message_hash = hashlib.sha256(message).digest()
    return private_key.sign(message_hash)

def verify_signature(public_key: bytes, message: bytes, signature: bytes):
    public_key_obj = VerifyingKey.from_string(public_key, curve=SECP256k1)
    message_hash = hashlib.sha256(message).digest()
    try:
        return public_key_obj.verify(signature, message_hash)
    except Exception:
        return False