from typing import Dict, List
from dataclasses import dataclass

from distributed_state_network.objects.endpoint import Endpoint

@dataclass(frozen=True)
class DSNodeConfig:
    node_id: str
    credential_dir: str
    port: int
    network_ip: str
    aes_key: str | None
    bootstrap_nodes: List[Endpoint]

    @staticmethod
    def from_dict(data: Dict) -> 'DSNodeConfig':
        return DSNodeConfig(
            data["node_id"], 
            data["credential_dir"] if "credential_dir" in data else "credentials",
            data["port"],
            data["network_ip"] if "network_ip" in data else "127.0.0.1", 
            data["aes_key"] if "aes_key" in data else None, 
            [Endpoint.from_json(e) for e in data["bootstrap_nodes"]]
        )
