import os
import logging
from typing import Callable, Tuple

from distributed_state_network.util.ecdsa import generate_key_pair

class KeyManager:
    def __init__(
        self,
        key_type: str,
        node_id: str, 
        folder: str,
        public_extension: str,
        private_extension: str,
        gen_keys: Callable[[], Tuple[bytes, bytes]]
    ):
        self.key_type = key_type
        self.node_id = node_id
        self.folder = folder
        self.public_extension = public_extension
        self.private_extension = private_extension
        self.gen_keys = gen_keys

    def write_public(self, node_id: str, cert: bytes):
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        if not os.path.exists(f'{self.folder}/{node_id}'):
            os.mkdir(f'{self.folder}/{node_id}')
        with open(f'{self.folder}/{self.node_id}/{node_id}.{self.public_extension}', 'wb') as f:
            f.write(cert)

    def read_public(self, node_id: str) -> bytes:
        if not os.path.exists(f'{self.folder}/{self.node_id}/{node_id}.{self.public_extension}'):
            raise Exception(401, f"Cannot find public ECDSA key for {node_id}")
        with open(f'{self.folder}/{self.node_id}/{node_id}.{self.public_extension}', 'rb') as f:
            return f.read()

    def has_public(self, node_id: str) -> bool:
        return os.path.exists(f'{self.folder}/{self.node_id}/{node_id}.{self.public_extension}')

    def ensure_public(self, node_id: str, public_key: bytes):
        if self.has_public(node_id):
            if not self.verify_public(node_id, public_key):
                raise Exception(401, f"Cannot verify ECDSA key for {node_id}")
        else:
            self.write_public(node_id, public_key)

    def public_path(self, node_id: str):
        return f"{self.folder}/{self.node_id}/{node_id}.{self.public_extension}"

    def verify_public(self, node_id: str, public_key: bytes):
        if not self.has_public(node_id):
            return False
        return public_key == self.read_public(node_id)

    def my_public(self) -> bytes:
        return self.read_public(self.node_id)

    def my_private(self):
        if not os.path.exists(f'{self.folder}/{self.node_id}/{self.node_id}.{self.private_extension}'):
            raise Exception("Private key not found")
        with open(f'{self.folder}/{self.node_id}/{self.node_id}.{self.private_extension}', 'rb') as f:
            return f.read()

    def generate_keys(self):
        if os.path.exists(f'{self.folder}/{self.node_id}/{self.node_id}.{self.private_extension}'):
            return
        logging.getLogger("DSN: " + self.node_id).info(f"Generating {self.key_type} Keys ...")
        cert_bytes, key_bytes = self.gen_keys()
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        if not os.path.exists(f'{self.folder}/{self.node_id}'):
            os.mkdir(f'{self.folder}/{self.node_id}')
        with open(f'{self.folder}/{self.node_id}/{self.node_id}.{self.public_extension}', 'wb') as f:
            f.write(cert_bytes)
        with open(f'{self.folder}/{self.node_id}/{self.node_id}.{self.private_extension}', 'wb') as f:
            f.write(key_bytes)

class CredentialManager(KeyManager):
    def __init__(self, folder: str, node_id: str):
        KeyManager.__init__(self, "ECDSA", node_id, folder, "pub", "key", generate_key_pair)
