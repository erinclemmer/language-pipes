from time import sleep
from pathlib import Path

from language_pipes.content_provider.content_provider import ContentProvider

class LpRunner:
    def __init__(self, config_file: Path):
        self.provider = ContentProvider(config_file)
        self.provider.network_provider.start_network()
        self.provider.job_provider.start_oai_server()
        for model in self.provider.model_provider.get_layer_models():
            self.provider.model_provider.host_layer_model(model)
        
        for model in self.provider.model_provider.get_end_models():
            self.provider.model_provider.host_end_model(model)

        self.log_output()
        
    def log_output(self):
        while True:
            # Consume all logs and wait for next loop
            status = self.provider.network_provider.get_network_status()
            if status is not None:
                for log in status.logs:
                    print(log)
            
            for log in self.provider.job_provider.get_oai_logs():
                print(log)
            
            for log in self.provider.model_provider.get_model_manager_logs():
                print(log)

            self.provider.network_provider.reset_router_logs()
            self.provider.job_provider.reset_oai_logs()
            self.provider.model_provider.reset_model_manager_logs()
            sleep(1)