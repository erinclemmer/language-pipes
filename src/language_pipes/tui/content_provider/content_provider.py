import psutil
from typing import Optional 

from language_pipes.util.utils import is_port_available
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.tui.content_provider.job_provider import JobProvider
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.pipe_provider import PipeProvider
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider

class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    pipe_manager: Optional[PipeManager]
    model_manager: ModelManager
    model_provider: ModelProvider
    network_provider: NetworkProvider
    pipe_provider: PipeProvider
    job_provider: JobProvider

    def __init__(self):
        self.router = None
        self.router_pipes = None
        self.pipe_manager = None
        self.model_manager = ModelManager()

        self.model_provider = ModelProvider(lambda: self.model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(lambda: self.router, self.set_router)
        self.pipe_provider = PipeProvider(lambda: self.pipe_manager)
        self.job_provider = JobProvider(lambda: self.router_pipes, lambda: self.model_manager, lambda: self.pipe_manager)

    def set_router(self, router: Optional[DSNodeServer]):
        self.router = router
        if router is not None:
            self.router_pipes = RouterPipes(router)
            self.pipe_manager = PipeManager(self.model_manager, self.router_pipes)
        else:
            self.router_pipes = None
            self.pipe_manager = None

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
            self.network_provider.stop_router()
            self.set_router(None)
        
        self.model_manager.stop()
        self.job_provider.stop_oai_server()