import os
from io import BytesIO

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, padding as sym_padding

def get_cipher(key: bytes, iv: bytes) -> Cipher:
    return Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

def get_iv() -> bytes:
    return os.urandom(16)

def generate_aes_key() -> bytes:
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=16,
        salt=os.urandom(16),
        iterations=100000,
        backend=default_backend()
    ).derive(os.urandom(128))
    iv = get_iv()
    bts = BytesIO()
    bts.write(iv)
    bts.write(key)
    return bts.getvalue()

def aes_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    bts = BytesIO(key)
    iv = bts.read(16)
    aes_key = bts.read()
    decryptor = get_cipher(aes_key, iv).decryptor()
    unpadder = sym_padding.PKCS7(128).unpadder()
    decrypted_text = decryptor.update(ciphertext) + decryptor.finalize()
    return unpadder.update(decrypted_text) + unpadder.finalize()

def aes_encrypt(key: bytes, data: bytes) -> bytes:
    bts = BytesIO(key)
    iv = bts.read(16)
    aes_key = bts.read()
    encryptor = get_cipher(aes_key, iv).encryptor()
    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    return encryptor.update(padded_data) + encryptor.finalize()
