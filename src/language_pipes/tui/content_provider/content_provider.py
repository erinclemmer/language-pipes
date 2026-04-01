from language_pipes.config import LpConfig
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider

class ContentProvider:
    router: DSNodeServer
    router_pipes: RouterPipes
    model_provider: ModelProvider
    network_provider: NetworkProvider

    def __init__(self, config: LpConfig):
        self.model_provider = ModelProvider()
        self.network_provider = NetworkProvider(self.get_router, self.set_router)

    def get_router(self):
        return self.router
    
    def set_router(self, router: DSNodeServer):
        self.router = router
        self.router_pipes = RouterPipes(router)