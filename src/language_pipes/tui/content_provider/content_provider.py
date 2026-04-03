from typing import Optional

import psutil
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider


class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    model_manager: ModelManager
    pipe_manager: PipeManager
    model_provider: ModelProvider
    network_provider: NetworkProvider

    def __init__(self):
        self.router = None
        self.router_pipes = None
        self.model_manager = ModelManager()
        self.model_provider = ModelProvider(self.get_model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(self.get_router, self.set_router)

    def get_router(self):
        return self.router

    def set_router(self, router: DSNodeServer):
        self.router = router
        self.router_pipes = RouterPipes(router) if router is not None else None
    
    def get_model_manager(self):
        return self.model_manager

    @staticmethod
    def get_total_system_ram() -> float:
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_used_system_ram() -> float:
        return psutil.virtual_memory().used / (1024**3)
