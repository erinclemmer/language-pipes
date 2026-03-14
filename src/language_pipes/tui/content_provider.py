import toml
from pathlib import Path
from language_pipes.util.config import default_config_dir
from language_pipes.distributed_state_network.objects.config import DSNodeConfig

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
    def save_network_config():
        pass # TODO