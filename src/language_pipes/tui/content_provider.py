import io
import os
import toml
import shutil
from tqdm.auto import tqdm
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict
from huggingface_hub import snapshot_download

from language_pipes.util.aes import generate_aes_key
from language_pipes.distributed_state_network.util import stop_thread
from language_pipes.util.config import default_config_dir, default_model_dir

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


class ModelDownloadProgress(tqdm):
    """tqdm subclass that silently captures progress for TUI display.

    The default tqdm renders the progress bar by writing to ``self.fp``
    (stderr) via ``display() -> sp() -> fp.write()``.  Overriding only
    ``write()`` is not enough because the bar itself never goes through
    ``write()``; it goes through ``display()``.

    To fully suppress terminal output we:
    1. Redirect ``file`` (``self.fp``) to ``os.devnull`` so any residual
       writes from the base class are silenced.
    2. Override ``display()`` to capture the formatted bar string into
       ``self.messages`` instead of printing it.
    3. Override ``clear()`` / ``close()`` so the base class never tries
       to erase lines on the real terminal.
    """

    latest_instance: Optional["ModelDownloadProgress"] = None

    def __init__(self, *args, **kwargs):
        if 'name' in kwargs:
            del kwargs['name']
        # Send fp to devnull so any base-class writes are harmless.
        kwargs['file'] = open(os.devnull, 'w')
        super().__init__(*args, **kwargs)
        self.messages: List[str] = []
        ModelDownloadProgress.latest_instance = self

    # -- output suppression / capture ------------------------------------------

    def display(self, msg=None, pos=None):  # type: ignore[override]
        """Capture the formatted bar string instead of printing it."""
        if msg is None:
            msg = str(self)
        if msg:
            self.messages.append(msg.strip())

    def clear(self, *args, **kwargs):
        """No-op – nothing to clear since we never wrote to the terminal."""
        pass

    def close(self):
        """Mark bar as disabled without writing anything to the terminal."""
        if self.disable:
            return
        self.disable = True
        lock = self.get_lock()
        with lock:
            type(self)._instances.discard(self)  # type: ignore[attr-defined]

    # -- public helpers --------------------------------------------------------

    def get_messages(self) -> List[str]:
        return list(self.messages)


class ContentProvider:
    router: Optional[DSNodeServer]
    router_starting: bool
    download_model_thread: Optional[Thread]
    router_thread: Optional[Thread]

    def __init__(self):
        self.router = None
        self.router_thread = None
        self.download_model_thread = None
        self.router_starting = False
    
    def start_download(self, model_id: str):
        if self.download_model_thread is not None:
            return
        clone_dir = Path(default_model_dir()) / model_id / "data"
        def download_model():
            snapshot_download(
                repo_id=model_id,
                local_dir=clone_dir,
                token=None,
                tqdm_class=ModelDownloadProgress
            )
        self.download_model_thread = Thread(target=download_model, args=())
        self.download_model_thread.start()

    def stop_model_download(self):
        if self.download_model_thread is None:
            return
        stop_thread(self.download_model_thread)
        self.download_model_thread = None

    def check_download_progress(self) -> Optional[List[str]]:
        if self.download_model_thread is None:
            return None
        if ModelDownloadProgress.latest_instance is None:
            return []
        return ModelDownloadProgress.latest_instance.get_messages()[-10:]

    # Network / Status
    def start_router(self, config_file: Path):
        if self.router_starting:
            return
        config = ContentProvider.get_network_config(config_file)
        def start_router():
            self.router_starting = True
            self.router = DSNodeServer.start(config)
            self.router_starting = False
        self.router_thread = Thread(target=start_router, args=())
        self.router_thread.start()

    def stop_router(self):
        if self.router_starting:
            return
        if self.router is None or self.router_thread is None:
            return
        self.router.stop()
        stop_thread(self.router_thread)
        self.router = None
        self.router_thread = None

    def get_router_status(self) -> Optional[RouterStatus]:
        if self.router is None and not self.router_starting:
            return None
        
        if self.router_starting:
            return RouterStatus(
                running=False,
                num_peers=0,
                logs=[]
            )

        if self.router is None:
            return None
        
        return RouterStatus(
            running=True,
            num_peers=len(self.router.node.node_states.keys()) - 1,
            logs=self.router.node.logs
        )
    
    # Network / Peers
    def get_peers(self) -> Dict[str, StatePacket]:
        if self.router is None:
            return { }
        data = self.router.node.node_states.copy()
        del data[self.router.node.config.node_id]
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
            network_ip=data.get("network_ip", ContentProvider.detect_network_ip()),
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

    # Models / Installed
    @staticmethod
    def get_installed_models() -> List[str]:
        models_dir = default_model_dir()

        models = []
        if not os.path.exists(models_dir):
            return models

        for org in os.listdir(models_dir):
            org_path = os.path.join(models_dir, org)
            if os.path.isdir(org_path):
                for model in os.listdir(org_path):
                    model_path = os.path.join(org_path, model)
                    if os.path.isdir(model_path):
                        models.append(f"{org}/{model}")

        return sorted(models)
    
    @staticmethod
    def delete_installed_model(model_name: str):
        model_dir = Path(default_model_dir()) / model_name
        if not os.path.exists(model_dir):
            return
        shutil.rmtree(model_dir)
        
