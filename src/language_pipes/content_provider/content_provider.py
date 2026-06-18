from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from typing import Callable, List, Optional, Dict

from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
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
    job_tracker: Optional[JobTracker]
    job_factory: Optional[JobFactory]
    job_receiver: Optional[JobReceiver]

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
        self.job_tracker = None
        self.job_factory = None
        self.job_receiver = None
        self.model_manager = ModelManager()
        self.config_file = config_file
        self.create_alert = create_alert
        self.state = ProviderState([], {})

        self.model_provider = ModelProvider(config_file, lambda: self.model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(config_file, lambda: self.router, self.set_router, self.create_alert)
        self.pipe_provider = PipeProvider(lambda: self.pipe_manager)
        self.job_provider = JobProvider(
            config_file, 
            lambda: self.router_pipes, 
            lambda: self.model_manager, 
            lambda: self.pipe_manager,
            lambda: self.job_tracker,
            lambda: self.job_factory,
            lambda: self.job_receiver
        )
        self.sync_provider_state()

    def sync_provider_state(self):
        self.state.visible_headers = ["Home", "Network", "Models"]
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
            model_paths.extend(["Layer Models", "End Models"])

        model_paths.append("Installed")

        self.state.visible_sub_menu["Models"] = model_paths

        if "Pipes" in self.state.visible_headers:
            self.state.visible_sub_menu["Pipes"] = ["Connected", "Complete", "Incomplete"]

        if "Jobs" in self.state.visible_headers:
            self.state.visible_sub_menu["Jobs"] = ["Server", "Active"]

    def set_router(self, router: Optional[DSNodeServer]):
        self.router = router
        if router is not None:
            self.router_pipes = RouterPipes(router)
            self.pipe_manager = PipeManager(self.model_manager, self.router_pipes)
            self.job_tracker = JobTracker()
            self.job_factory = JobFactory(self.job_tracker, self.pipe_manager)
            self.job_receiver = JobReceiver(
                job_factory=self.job_factory,
                job_tracker=self.job_tracker,
                model_manager=self.model_manager,
                pipe_manager=self.pipe_manager,
                is_shutdown=self.router_pipes.router.is_shut_down
            )

            self.router_pipes.router.set_receive_cb(self.job_receiver.receive_data)
        else:
            self.router_pipes = None
            self.pipe_manager = None

    def stop_network(self):
        if self.router is None:
            return
        self.model_provider.unload_all_models()
        self.job_provider.stop_oai_server()
        self.network_provider._stop_network()
        if self.job_tracker is not None:
            self.job_tracker.shutdown = True
        if self.job_receiver is not None:
            self.job_receiver.shutdown = True
        

    @staticmethod
    def get_total_system_ram() -> float:
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_used_system_ram() -> float:
        return psutil.virtual_memory().used / (1024**3)
    
    @staticmethod
    def get_total_swap() -> float:
        return psutil.swap_memory().total / (1024**3)
    
    @staticmethod
    def get_used_swap() -> float:
        return psutil.swap_memory().used / (1024**3)

    @staticmethod
    def get_cuda_device_count() -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    @staticmethod
    def get_total_cuda_memory(device: int) -> float:
        _, total = torch.cuda.mem_get_info(device)
        return total / (1024**3)

    @staticmethod
    def get_used_cuda_memory(device: int) -> float:
        free, total = torch.cuda.mem_get_info(torch.device(f'cuda:{device}'))
        return (total - free) / (1024**3)

    @staticmethod
    def get_ram_usage() -> str:
        used_ram = ContentProvider.get_used_system_ram()
        total_ram = ContentProvider.get_total_system_ram()

        used_swap = ContentProvider.get_used_swap()
        total_swap = ContentProvider.get_total_swap()
        
        return f"System RAM:  {used_ram:.1f}/{total_ram:.1f}GB".ljust(26) + f"System Swap: {used_swap:.1f}/{total_swap:.1f}GB"

    @staticmethod
    def is_port_available(port: int) -> bool:
        return is_port_available(port)
    
    def shutdown(self):
        if self.router is not None:
            self.stop_network()
            self.set_router(None)
        
        self.model_manager.stop()
        self.job_provider.stop_oai_server()