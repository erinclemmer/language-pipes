import logging
from time import sleep
from threading import Event
from pathlib import Path

from language_pipes.config import LpConfig
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.util.config import initialize_folders

logger = logging.getLogger(__name__)

class LpRunner:
    def __init__(self, config_file: Path):
        initialize_folders()
        
        config = LpConfig.from_file(config_file)

        # Headless has no alert popup, so surface alerts through the console
        # handler the CLI installed.
        def create_alert(alert: str):
            logger.warning(alert)

        self.provider = ContentProvider(config_file, create_alert)

        generated_node_ids = self.provider.network_provider.get_my_node_ids()
        if config.network_config.node_id not in generated_node_ids:
            self.provider.network_provider.save_new_node_id(config.network_config.node_id)

        self.provider.network_provider.start_network(config.network_config)

        while self.provider.network_provider.router_starting:
            sleep(0.1)

        self.provider.job_provider.start_oai_server(config)

        for model in config.layer_models:
            self.provider.model_provider.load_layer_model(model)

        for model in config.end_models:
            self.provider.model_provider.load_end_model(model)

        self.wait()

    def wait(self):
        """Block forever; log records reach stdout via the console handler."""
        try:
            Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down")
