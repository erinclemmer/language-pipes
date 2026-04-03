from typing import Optional
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider


class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    model_manager: ModelManager
    model_provider: ModelProvider
    network_provider: NetworkProvider

    def __init__(self):
        self.router = None
        self.router_pipes = None
        self.model_manager = ModelManager()
        self.model_provider = ModelProvider(self.model_manager)
        self.network_provider = NetworkProvider(self.get_router, self.set_router)

    def get_router(self):
        return self.router

    def set_router(self, router: DSNodeServer):
        self.router = router
        self.router_pipes = RouterPipes(router)
        self.model_provider.set_router_pipes(self.router_pipes)
