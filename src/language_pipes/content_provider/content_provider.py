from dataclasses import dataclass
from pathlib import Path

import psutil
from typing import Callable, List, Optional, Dict 

from language_pipes.util.utils import is_port_available
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.content_provider.job_provider import JobProvider
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.content_provider.pipe_provider import PipeProvider
from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.content_provider.network_provider import NetworkProvider

@dataclass
class ProviderState:
    visible_headers: List[str]
    visible_sub_menu: Dict[str, List[str]]

class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    pipe_manager: Optional[PipeManager]
    model_manager: ModelManager
    model_provider: ModelProvider
    network_provider: NetworkProvider
    pipe_provider: PipeProvider
    job_provider: JobProvider
    config_file: Path
    create_alert: Callable[[str], None]

    def __init__(self, config_file: Path, create_alert: Callable[[str], None]):
        self.router = None
        self.router_pipes = None
        self.pipe_manager = None
        self.model_manager = ModelManager()
        self.config_file = config_file
        self.create_alert = create_alert
        self.state = ProviderState([], {})

        self.model_provider = ModelProvider(config_file, lambda: self.model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(config_file, lambda: self.router, self.set_router, self.create_alert)
        self.pipe_provider = PipeProvider(lambda: self.pipe_manager)
        self.job_provider = JobProvider(config_file, lambda: self.router_pipes, lambda: self.model_manager, lambda: self.pipe_manager)
        self.sync_provider_state()

    def sync_provider_state(self):
        self.state.visible_headers = ["Home", "Network", "Models", "Pipes", "Jobs"]
        if self.router is not None and self.router.running:
            self.state.visible_headers.extend(["Pipes", "Jobs"])

        self.state.visible_sub_menu = {
            "Home": ["Dashboard", "Activity"]
        }

        network_paths = []
        network_config = self.network_provider.get_network_config()
        if network_config.node_id is not None:
            network_paths.append("Status")
        
        if self.router is not None and self.router.running:
            network_paths.append("Peers")
        
        network_paths.append("Configure")

        self.state.visible_sub_menu["Network"] = network_paths

        model_paths = []

        if self.router is not None and self.router.running:
            model_paths.append("Hosted")

        model_paths.append("Installed")

        self.state.visible_sub_menu["Models"] = model_paths

    def set_router(self, router: Optional[DSNodeServer]):
        self.router = router
        if router is not None:
            self.router_pipes = RouterPipes(router)
            self.pipe_manager = PipeManager(self.model_manager, self.router_pipes)
        else:
            self.router_pipes = None
            self.pipe_manager = None

    def stop_network(self):
        if self.router is None:
            return
        self.model_provider.unload_all_models()
        self.job_provider.stop_oai_server()
        self.network_provider._stop_network()

    @staticmethod
    def get_total_system_ram() -> float:
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_used_system_ram() -> float:
        return psutil.virtual_memory().used / (1024**3)
    
    @staticmethod
    def is_port_available(port: int) -> bool:
        return is_port_available(port)
    
    def shutdown(self):
        if self.router is not None:
            self.stop_network()
            self.set_router(None)
        
        self.model_manager.stop()
        self.job_provider.stop_oai_server()