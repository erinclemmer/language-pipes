import os
import shutil
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Tuple

from language_pipes.config import LpConfig
from language_pipes.util.config import get_app_dir
from language_pipes.util.aes import generate_aes_key
from language_pipes.distributed_state_network.util import stop_thread

from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.state_packet import StatePacket
from language_pipes.distributed_state_network.util.key_manager import CredentialManager

AES_KEY_LEN = 16

@dataclass
class RouterStatus:
    state: str
    running: bool
    node_id: str
    num_peers: int
    logs: List[Tuple[float, str]]
    port: int
    encrypted: bool

class NetworkProvider:
    router_thread: Optional[Thread]
    router_starting: bool
    router_stopping: bool
    config_file: Path

    get_router: Callable[[], Optional[DSNodeServer]]
    create_alert: Callable[[str], None]

    def __init__(
        self, 
        config_file: Path, 
        get_router: Callable, 
        set_router: Callable,
        create_alert: Callable[[str], None]
    ):
        self.router_starting = False
        self.router_stopping = False
        self.router_thread = None
        self.config_file = config_file
        self.get_router = get_router
        self.set_router = set_router
        self.create_alert = create_alert
        self.set_router(None)

    # Network / Status
    def start_network(self, config: Optional[DSNodeConfig] = None):
        if self.router_starting or self.router_stopping:
            return
        
        if config is None:
            config = self.get_network_config() 
        
        if config.node_id is None:
            return
        
        if config.aes_key is not None and not config.aes_key_is_valid():
            return
        
        self.router_starting = True
        def start_router():
            self.set_router(DSNodeServer.start(config, self.create_alert))
            self.router_starting = False
        self.router_thread = Thread(target=start_router, args=())
        self.router_thread.start()

    def _stop_network(self):
        if self.router_starting or self.router_stopping:
            return
        rtr = self.get_router()
        router_thread = self.router_thread
        if rtr is None or router_thread is None:
            return
        self.router_stopping = True
        def stop_router():
            try:
                rtr.stop()
                stop_thread(router_thread)
                self.set_router(None)
                self.router_thread = None
            finally:
                self.router_stopping = False
        Thread(target=stop_router, args=()).start()

    def get_network_status(self) -> Optional[RouterStatus]:
        rtr = self.get_router()
        if rtr is None: 
            return None
        if rtr is None and not self.router_starting and not self.router_stopping:
            return None
        
        if self.router_starting:
            return RouterStatus(
                node_id=rtr.node_id(),
                state="starting",
                running=False,
                num_peers=0,
                logs=[],
                port=rtr.config.port,
                encrypted=False
            )

        if self.router_stopping:
            return RouterStatus(
                node_id=rtr.node_id(),
                state="stopping",
                running=False,
                num_peers=0 if rtr is None else len(rtr.node.node_states.keys()) - 1,
                logs=[] if rtr is None else rtr.node.logs,
                port=rtr.config.port,
                encrypted=False
            )
        
        return RouterStatus(
            node_id=rtr.node_id(),
            state="running",
            running=True,
            num_peers=len(rtr.node.node_states.keys()) - 1,
            logs=rtr.node.logs,
            port=rtr.config.port,
            encrypted=rtr.config.aes_key is not None
        )
    
    def reset_router_logs(self):
        rtr = self.get_router()
        if rtr is None:
            return
        rtr.node.logs = []

    # Network / Peers
    def get_peers(self) -> Dict[str, StatePacket]:
        rtr = self.get_router()
        if rtr is None:
            return { }
        data = rtr.node.node_states.copy()
        del data[rtr.node.config.node_id]
        return data

    # Network / Configure
    def get_network_config(self) -> DSNodeConfig:
        cfg = LpConfig.from_file(self.config_file).network_config

        if cfg.network_ip is None:
            cfg.network_ip = NetworkProvider.detect_network_ip()
        
        return cfg
    
    def save_network_config(self, config: DSNodeConfig):
        cfg = LpConfig.from_file(self.config_file)
        cfg.network_config = config
        cfg.save()

    @staticmethod
    def get_registered_node_ids() -> List[str]:
        cred_dir = get_app_dir() / "credentials"
        if not os.path.exists(cred_dir):
            return []
        return os.listdir(cred_dir)
    
    @staticmethod
    def delete_node_id(node_id: str):
        cred_dir = get_app_dir() / "credentials"
        path = cred_dir / node_id
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def save_new_node_id(node_id: str):
        cred_dir = get_app_dir() / "credentials"
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