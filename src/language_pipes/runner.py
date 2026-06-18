from time import sleep, time
from datetime import datetime
from pathlib import Path
from typing import List

from language_pipes.config import LpConfig
from language_pipes.content_provider.content_provider import ContentProvider

class LpRunner:
    alerts: List[str]

    def __init__(self, config_file: Path):
        config = LpConfig.from_file(config_file)
        self.alerts = []

        def create_alert(alert: str):
            self.alerts.append(alert)

        self.provider = ContentProvider(config_file, create_alert)

        self.provider.network_provider.start_network(config.network_config)
        
        while self.provider.network_provider.router_starting:
            sleep(0.1)

        self.provider.job_provider.start_oai_server(config)

        for model in config.layer_models:
            self.provider.model_provider.load_layer_model(model)
        
        for model in config.end_models:
            self.provider.model_provider.load_end_model(model)

        self.log_output()
    
    def _format_time(self, t: float) -> str:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    def log_output(self):
        while True:
            # Consume all logs and wait for next loop
            status = self.provider.network_provider.get_network_status()
            if status is not None:
                for t, log in status.logs:
                    print(f"{self._format_time(t)}: {log}")
            
            for t, log in self.provider.job_provider.get_oai_logs():
                print(f"{self._format_time(t)}: {log}")
            
            for t, log in self.provider.model_provider.get_model_manager_logs():
                print(f"{self._format_time(t)}: {log}")

            if self.provider.job_factory is not None:
                for t, log in self.provider.job_factory.logs:
                    print(f"{self._format_time(t)}: {log}")

            if self.provider.job_receiver is not None:
                for t, log in self.provider.job_receiver.logs:
                    print(f"{self._format_time(t)}: {log}")

            for alert in self.alerts:
                print(f"{self._format_time(time())}: {alert}")

            self.alerts = []
            self.provider.network_provider.reset_router_logs()
            self.provider.job_provider.reset_oai_logs()
            self.provider.model_provider.reset_model_manager_logs()
            sleep(1)