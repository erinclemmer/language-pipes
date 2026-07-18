import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, padding as sym_padding


AES_KEY_LENGTH = 16
AES_BLOCK_SIZE = 16

def get_cipher(key: bytes, iv: bytes) -> Cipher:
    return Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

def get_iv() -> bytes:
    return os.urandom(AES_BLOCK_SIZE)


def _validate_aes_key(key: bytes):
    if len(key) != AES_KEY_LENGTH:
        raise ValueError(
            f"Invalid AES key length ({len(key)} bytes). Expected {AES_KEY_LENGTH} bytes (AES-128 key only)."
        )

def generate_aes_key() -> bytes:
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_LENGTH,
        salt=os.urandom(AES_BLOCK_SIZE),
        iterations=100000,
        backend=default_backend()
    ).derive(os.urandom(128))
    return key

def aes_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    _validate_aes_key(key)
    if len(ciphertext) < AES_BLOCK_SIZE:
        raise ValueError("Ciphertext too short. Expected IV prefix followed by ciphertext.")

    iv = ciphertext[:AES_BLOCK_SIZE]
    actual_ciphertext = ciphertext[AES_BLOCK_SIZE:]
    decryptor = get_cipher(key, iv).decryptor()
    unpadder = sym_padding.PKCS7(128).unpadder()
    decrypted_text = decryptor.update(actual_ciphertext) + decryptor.finalize()
    return unpadder.update(decrypted_text) + unpadder.finalize()

def aes_encrypt(key: bytes, data: bytes) -> bytes:
    _validate_aes_key(key)
    iv = get_iv()
    encryptor = get_cipher(key, iv).encryptor()
    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext
