import os
import toml
import shutil
from typing import List
from pathlib import Path

from language_pipes.util.aes import generate_aes_key
from language_pipes.util.config import default_config_dir
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint
from language_pipes.distributed_state_network.util.key_manager import CredentialManager

AES_KEY_LEN = 32

class ContentProvider:
    @staticmethod
    def get_network_config(config_file: Path) -> DSNodeConfig:
        with open(config_file, 'r') as f:
            data = toml.load(f)
        return DSNodeConfig(
            node_id=data.get("node_id", ""),
            credential_dir=default_config_dir() + "/credentials",
            port=data.get("peer_port", 5000),
            network_ip=data.get("network_ip", ContentProvider.detect_network_ip()),
            aes_key=data.get("aes_key", None),
            bootstrap_nodes=[Endpoint(d["address"], int(d["port"])) for d in  data.get("bootstrap_nodes", [])],
            whitelist_ips=[],
            whitelist_node_ids=[]
        )
    
    @staticmethod
    def save_network_config(save_file: Path, config: DSNodeConfig):
        data = { }
        if os.path.exists(save_file):
            data = toml.loads(save_file.read_text())
        data["node_id"] = config.node_id
        data["aes_key"] = config.aes_key
        if len(config.bootstrap_nodes) > 0:
            data["bootstrap_nodes"] = [{
                "address": n.address,
                "port": n.port
            } for n in config.bootstrap_nodes]
        with open(save_file, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

    @staticmethod
    def get_registered_node_ids() -> List[str]:
        cred_dir = default_config_dir() + "/credentials"
        if not os.path.exists(cred_dir):
            return []
        return os.listdir(cred_dir)
    
    @staticmethod
    def delete_node_id(node_id: str):
        cred_dir = default_config_dir() + "/credentials"
        path = cred_dir + "/" + node_id
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def save_new_node_id(node_id: str):
        cred_dir = default_config_dir() + "/credentials"
        cred_manager = CredentialManager(cred_dir, node_id)
        cred_manager.generate_keys()

    @staticmethod
    def generate_aes_key() -> str:
        return generate_aes_key().hex()

    @staticmethod
    def validate_aes_key(key: str) -> bool:
        try:
            bts = bytes.fromhex(key)
            return len(bts) == AES_KEY_LEN
        except Exception:
            return False
        
    @staticmethod
    def detect_network_ip() -> str:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()