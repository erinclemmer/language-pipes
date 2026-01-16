import os
from io import BytesIO
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

def generate_aes_key() -> bytes:
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=16,
        salt=os.urandom(16),
        iterations=100000,
        backend=default_backend()
    ).derive(os.urandom(128))
    iv = os.urandom(16)
    bts = BytesIO()
    bts.write(iv)
    bts.write(key)
    return bts.getvalue()

def save_new_aes_key(file_path: str) -> str:
    key = generate_aes_key()
    key_hex = key.hex()
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(key_hex)
    return key_hex
