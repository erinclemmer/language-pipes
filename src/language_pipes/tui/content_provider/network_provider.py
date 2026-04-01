import os
import toml
import shutil
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable

from language_pipes.util.aes import generate_aes_key
from language_pipes.distributed_state_network.util import stop_thread
from language_pipes.util.config import default_config_dir

from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.state_packet import StatePacket
from language_pipes.distributed_state_network.objects.endpoint import Endpoint
from language_pipes.distributed_state_network.util.key_manager import CredentialManager

AES_KEY_LEN = 32


@dataclass
class RouterStatus:
    running: bool
    num_peers: int
    logs: List

class NetworkProvider:
    router_thread: Optional[Thread]
    router_starting: bool

    def __init__(self, get_router: Callable, set_router: Callable):
        self.router_starting = False
        self.router_thread = None
        self.get_router = get_router
        self.set_router = set_router
        self.set_router(None)

    # Network / Status
    def start_router(self, config_file: Path):
        if self.router_starting:
            return
        config = NetworkProvider.get_network_config(config_file)
        def start_router():
            self.router_starting = True
            self.set_router(DSNodeServer.start(config))
            self.router_starting = False
        self.router_thread = Thread(target=start_router, args=())
        self.router_thread.start()

    def stop_router(self):
        if self.router_starting:
            return
        rtr = self.get_router()
        if rtr is None or self.router_thread is None:
            return
        rtr.stop()
        stop_thread(self.router_thread)
        self.set_router(None)
        self.router_thread = None

    def get_router_status(self) -> Optional[RouterStatus]:
        rtr = self.get_router()
        if rtr is None and not self.router_starting:
            return None
        
        if self.router_starting:
            return RouterStatus(
                running=False,
                num_peers=0,
                logs=[]
            )

        if rtr is None:
            return None
        
        return RouterStatus(
            running=True,
            num_peers=len(rtr.node.node_states.keys()) - 1,
            logs=rtr.node.logs
        )
    
    # Network / Peers
    def get_peers(self) -> Dict[str, StatePacket]:
        rtr = self.get_router()
        if rtr is None:
            return { }
        data = rtr.node.node_states.copy()
        del data[rtr.node.config.node_id]
        return data

    # Network / Configure
    @staticmethod
    def get_network_config(config_file: Path) -> DSNodeConfig:
        with open(config_file, 'r') as f:
            data = toml.load(f)
        return DSNodeConfig(
            node_id=data.get("node_id", ""),
            credential_dir=default_config_dir() + "/credentials",
            logging_dir=default_config_dir() + "/logs",
            port=data.get("peer_port", 5000),
            network_ip=data.get("network_ip", NetworkProvider.detect_network_ip()),
            aes_key=data.get("aes_key", None),
            bootstrap_nodes=[Endpoint(d["address"], int(d["port"])) for d in  data.get("bootstrap_nodes", [])],
            whitelist_ips=[],
            whitelist_node_ids=data.get("whitelist_node_ids", [])
        )
    
    @staticmethod
    def save_network_config(save_file: Path, config: DSNodeConfig):
        data = { }
        if os.path.exists(save_file):
            data = toml.loads(save_file.read_text())
        data["node_id"] = config.node_id
        data["aes_key"] = config.aes_key
        data["network_ip"] = config.network_ip
        data["peer_port"] = config.port
        data["bootstrap_nodes"] = [{
            "address": n.address,
            "port": n.port
        } for n in config.bootstrap_nodes]
        data["whitelist_node_ids"] = config.whitelist_node_ids
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