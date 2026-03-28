from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider

class ContentProvider:
    model_provider: ModelProvider
    network_provider: NetworkProvider

    def __init__(self):
        self.model_provider = ModelProvider()
        self.network_provider = NetworkProvider()