import os
import toml
import shutil
from typing import List
from pathlib import Path

from language_pipes.util.config import default_config_dir
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.util.key_manager import CredentialManager

class ContentProvider:
    @staticmethod
    def get_network_config(config_file: Path) -> DSNodeConfig:
        with open(config_file, 'r') as f:
            data = toml.load(f)
        return DSNodeConfig(
            node_id=data.get("node_id", ""),
            credential_dir=default_config_dir() + "/credentials",
            port=data.get("peer_port", 5000),
            network_ip=data.get("network_ip", "127.0.0.1"),
            aes_key=data.get("aes_key", None),
            bootstrap_nodes=data.get("bootstrap_nodes", []),
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
                "address": config.bootstrap_nodes[0].address,
                "port": config.bootstrap_nodes[0].port
            }]
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