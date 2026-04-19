from pathlib import Path

from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.content_provider.providers import get_providers

class LpRunner:
    def __init__(self, config_file: Path):
        self.loader = ContentLoader(get_providers(config_file))
        self.loader.call_provider(ProviderCall.start_network)
        self.loader.call_provider(ProviderCall.start_oai_server)
        for model in self.loader.call_provider(ProviderCall.get_layer_models):
            self.loader.call_provider(ProviderCall.host_layer_model, model.model_id)
        
        for model in self.loader.call_provider(ProviderCall.get_end_models):
            self.loader.call_provider(ProviderCall.host_end_model, model)