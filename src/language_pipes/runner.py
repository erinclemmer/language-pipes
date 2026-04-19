from time import sleep
from pathlib import Path
from typing import Optional

from language_pipes.content_loader import ContentLoader
from language_pipes.content_provider.providers import get_providers
from language_pipes.content_provider.network_provider import RouterStatus
from language_pipes.content_provider.provider_calls import ProviderCall

class LpRunner:
    def __init__(self, config_file: Path):
        self.loader = ContentLoader(get_providers(config_file))
        self.loader.call_provider(ProviderCall.start_network)
        self.loader.call_provider(ProviderCall.start_oai_server)
        for model in self.loader.call_provider(ProviderCall.get_layer_models):
            self.loader.call_provider(ProviderCall.host_layer_model, model.model_id)
        
        for model in self.loader.call_provider(ProviderCall.get_end_models):
            self.loader.call_provider(ProviderCall.host_end_model, model)

        self.log_output()
        
    def log_output(self):
        while True:
            # Consume all logs and wait for next loop
            status: Optional[RouterStatus] = self.loader.call_provider(ProviderCall.get_network_status)
            if status is not None:
                for log in status.logs:
                    print(log)
                self.loader.call_provider(ProviderCall.reset_router_logs)
            
            for log in self.loader.call_provider(ProviderCall.get_oai_logs):
                print(log)
            
            self.loader.call_provider(ProviderCall.reset_oai_logs)

            for log in self.loader.call_provider(ProviderCall.get_model_manager_logs):
                print(log)

            self.loader.call_provider(ProviderCall.reset_model_manager_logs)
            sleep(1)