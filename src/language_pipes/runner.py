import logging
from time import sleep
from threading import Event
from pathlib import Path
from typing import Optional

from language_pipes.config import LpConfig
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.util.config import initialize_folders

logger = logging.getLogger(__name__)

class LpRunner:
    def __init__(self, config_file: Path, hf_token: Optional[str] = None):
        initialize_folders()
        
        config = LpConfig.from_file(config_file)

        # Headless has no alert popup, so surface alerts through the console
        # handler the CLI installed.
        def create_alert(alert: str):
            logger.warning(alert)

        self.provider = ContentProvider(config_file, create_alert)

        self._generate_node_id(config)
        self._download_models(config, hf_token)

        self.provider.network_provider.start_network(config.network_config)

        while self.provider.network_provider.router_starting:
            sleep(0.1)

        self.provider.job_provider.start_oai_server(config)

        for model in config.layer_models:
            self.provider.model_provider.load_layer_model(model)

        for model in config.end_models:
            self.provider.model_provider.load_end_model(model)

        self.wait()

    def _generate_node_id(self, config: LpConfig):
        generated_node_ids = self.provider.network_provider.get_my_node_ids()
        if config.network_config.node_id not in generated_node_ids:
            self.provider.network_provider.save_new_node_id(config.network_config.node_id)

    def _download_models(self, config: LpConfig, token: Optional[str]):
        if token is None:
            token = self.provider.model_provider.get_hf_config_token()
        
        downloaded_models = self.provider.model_provider.get_installed_models()
        for m in config.layer_models:
            if m.model_id not in downloaded_models:
                self._download_model(m.model_id, token)
                downloaded_models.append(m.model_id)

        for m in config.end_models:
            if m not in downloaded_models:
                self._download_model(m, token)
                downloaded_models.append(m)

    def _download_model(self, model_id: str, token: Optional[str]):
        """Download one model, blocking until it finishes.

        start_download runs in a thread and no-ops while another download is
        active, so the thread must be joined before starting the next model.
        """
        logger.info(f"Downloading {model_id}...")
        self.provider.model_provider.start_download(model_id, token)

        thread = self.provider.model_provider.download_model_thread
        if thread is not None:
            thread.join()

        message = self.provider.model_provider.download_message
        if message is not None and "[ERROR]" in message:
            logger.error(f"Failed to download {model_id}: {message}")
            return

        logger.info("Download complete!")


    def wait(self):
        """Block forever; log records reach stdout via the console handler."""
        try:
            Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down")
