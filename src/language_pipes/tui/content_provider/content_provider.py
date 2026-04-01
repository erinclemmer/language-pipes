from language_pipes.lp import LanguagePipes
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider

class ContentProvider:
    router: DSNodeServer
    model_provider: ModelProvider
    network_provider: NetworkProvider

    def __init__(self):
        self.model_provider = ModelProvider()
        self.network_provider = NetworkProvider(self.get_router, self.set_router)

    def get_router(self):
        return self.router
    
    def set_router(self, router: DSNodeServer):
        self.router = router