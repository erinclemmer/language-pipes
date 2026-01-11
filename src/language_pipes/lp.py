import logging
from threading import Thread

from language_pipes.oai_server import OAIHttpServer

from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker

from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.modeling.model_manager import ModelManager

from language_pipes.util import stop_thread
from language_pipes.config import LpConfig
from language_pipes.network import StateNetworkServer

class LanguagePipes:
    router: StateNetworkServer
    
    job_factory: JobFactory
    job_receiver: JobReceiver

    oai_server: OAIHttpServer
    oai_thread: Thread
    
    config: LpConfig

    def __init__(
        self, 
        version: str,
        config: LpConfig,
        router: StateNetworkServer
    ):
        self.config = config
        self.set_logging_level(self.config.logging_level)
        
        self.router_pipes = None
        self.router = router
        logger = self.router.node.logger

        # Network pipe data for MetaPipe objects
        self.router_pipes = RouterPipes(self.router.node)

        # Local pipe data for LlmModel objects
        self.model_manager = ModelManager(
            logger=logger,
            config=self.config,
            # Used for placing model data on the network
            router_pipes=self.router_pipes
        )

        # Merge local and network data to get Pipe object
        self.pipe_manager = PipeManager(
            config=self.config,
            model_manager=self.model_manager,
            router_pipes=self.router_pipes
        )

        # View currently loaded pipes
        self.router_pipes.print_pipes()

        # Holds pending jobs
        self.job_tracker = JobTracker(logger, self.config)

        # Handles job creation
        self.job_factory = JobFactory(
            logger=logger,
            config=self.config, 
            job_tracker=self.job_tracker,
            get_pipe_by_model_id=self.pipe_manager.get_pipe_by_model_id,
        )

        is_shutdown = lambda: self.router.node.shutting_down

        # Receives jobs and creates JobProcessor object before processing
        self.job_receiver = JobReceiver(
            logger=logger, 
            config=self.config, 
            job_tracker=self.job_tracker,
            job_factory=self.job_factory,
            pipe_manager=self.pipe_manager,
            model_manager=self.model_manager,
            is_shutdown=is_shutdown
        )

        # Broadcast our job port for other nodes to connect to
        self.router.node.update_data("job_port", str(self.config.job_port))

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
        self.oai_server = OAIHttpServer(self.config.oai_port, self.job_factory.start_job)
        self.oai_thread = Thread(target=self.oai_server.serve_forever, args=())
        self.oai_thread.start()
        self.job_factory.logger.info(f"OpenAI Server started on port {self.config.oai_port}")

    def set_logging_level(self, logging_level: str):
        level = getattr(logging, logging_level.upper(), None)
        if level is None:
            raise ValueError(f"Invalid logging level: {logging_level}")
        logging.basicConfig(level=level)

    def router_port(self) -> int:
        return int(self.router.node.port)

    def stop(self):
        self.model_manager.stop()
        self.job_receiver.stop()
        self.router.stop()
        if self.config.oai_port is not None:
            self.oai_server.shutdown()
            stop_thread(self.oai_thread)
