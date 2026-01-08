import logging
from threading import Thread
from distributed_state_network import DSNodeServer

from language_pipes.oai_server import OAIHttpServer
from language_pipes.jobs.job_manager import JobManager
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager

from language_pipes.util import stop_thread
from language_pipes.config import LpConfig

def serve(httpd):
    httpd.serve_forever()

class LanguagePipes:
    router: DSNodeServer
    
    job_manager: JobManager
    job_receiver: JobReceiver

    oai_server: OAIHttpServer
    oai_thread: Thread
    
    config: LpConfig

    def __init__(
        self, 
        version: str,
        config: LpConfig
    ):
        self.config = config
        self.set_logging_level(self.config.logging_level, self.config.router.node_id)
        
        self.router_pipes = None
        self.router = DSNodeServer.start(self.config.router, self.print_pipes, self.print_pipes)

        self.router_pipes = RouterPipes(self.router.node)

        self.model_manager = ModelManager(
            config.router.node_id,
            config.app_dir,
            self.router_pipes,
            self.router.node.logger,
            self.config.processor
        )

        self.job_tracker = JobTracker(self.router.node.logger)

        self.job_manager = JobManager(
            app_dir=config.app_dir, 
            router=self.router.node, 
            config=self.config.processor, 
            router_pipes=self.router_pipes,
            job_tracker=self.job_tracker,
            get_layer_models=self.model_manager.get_layer_models,
            get_end_model=self.model_manager.get_end_model
        )

        self.job_receiver = JobReceiver(
            config=self.config.processor, 
            router=self.router.node, 
            job_tracker=self.job_tracker,
            job_manager=self.job_manager,
            get_end_model=self.model_manager.get_end_model
        )

        if self.config.oai_port is not None:
            self.start_oai()

    def print_pipes(self):
        if self.router_pipes is None:
            return
        self.router_pipes.print_pipes()

    def start_oai(self):
        if self.config.oai_port is None:
            self.router.logger.error("Tried to start Open AI server but no port was specified")
            return
        self.oai_server = OAIHttpServer(self.config.oai_port, self.job_manager.start_job)
        self.oai_thread = Thread(target=self.oai_server.serve_forever, args=())
        self.oai_thread.start()
        self.job_manager.logger.info(f"OpenAI Server started on port {self.config.oai_port}")

    def set_logging_level(self, logging_level: str, router_id: str):
        level = getattr(logging, logging_level.upper(), None)
        if level is None:
            raise ValueError(f"Invalid logging level: {logging_level}")
        logging.basicConfig(level=level)

    def router_port(self) -> int:
        return self.config.router.port

    def stop(self):
        self.model_manager.stop()
        self.job_receiver.stop()
        self.router.stop()
        if self.config.oai_port is not None:
            self.oai_server.shutdown()
            stop_thread(self.oai_thread)
