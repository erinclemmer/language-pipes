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
from language_pipes.network_protocol import StateNetworkNode

class LanguagePipes:
    logger: logging.Logger
    router: StateNetworkNode
    
    job_factory: JobFactory
    job_receiver: JobReceiver

    oai_server: OAIHttpServer
    oai_thread: Thread
    
    config: LpConfig

    def __init__(
        self, 
        config: LpConfig,
        router: StateNetworkNode
    ):
        self.config = config
        router.set_receive_cb(self.receive_data)
        router.set_update_cb(self.print_pipes)
        router.set_disconnect_cb(self.print_pipes)

        self.logger = logging.getLogger(f"LP-{self.config.node_id}")
        self.set_logging_level(self.config.logging_level)
        
        self.router_pipes = None
        self.router = router

        # Network pipe data for MetaPipe objects
        self.router_pipes = RouterPipes(router)

        # Local pipe data for LlmModel objects
        self.model_manager = ModelManager(
            logger=self.logger,
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
        self.router_pipes.print_pipes(self.logger)

        # Holds pending jobs
        self.job_tracker = JobTracker(self.logger, self.config)

        # Handles job creation
        self.job_factory = JobFactory(
            logger=self.logger,
            config=self.config, 
            job_tracker=self.job_tracker,
            pipe_manager=self.pipe_manager
        )

        # Receives jobs and creates JobProcessor object before processing
        self.job_receiver = JobReceiver(
            logger=self.logger, 
            config=self.config, 
            job_tracker=self.job_tracker,
            job_factory=self.job_factory,
            pipe_manager=self.pipe_manager,
            model_manager=self.model_manager,
            is_shutdown=router.is_shut_down
        )

        if self.config.oai_port is not None:
            self.start_oai()

    def receive_data(self, data: bytes):
        self.job_receiver.receive_data(data)

    def print_pipes(self):
        if self.router_pipes is None:
            return
        self.router_pipes.print_pipes(self.logger)

    def start_oai(self):
        if self.config.oai_port is None:
            self.logger.error("Tried to start Open AI server but no port was specified")
            return
        self.oai_server = OAIHttpServer(self.config.oai_port, self.job_factory.start_job, self.router_pipes.get_models)
        self.oai_thread = Thread(target=self.oai_server.serve_forever, args=())
        self.oai_thread.start()
        self.job_factory.logger.info(f"OpenAI Server started on port {self.config.oai_port}")

    def set_logging_level(self, logging_level: str):
        level = getattr(logging, logging_level.upper(), None)
        if level is None:
            raise ValueError(f"Invalid logging level: {logging_level}")
        logging.basicConfig(level=level)

    def stop(self):
        self.model_manager.stop()
        self.router.stop()
        if self.config.oai_port is not None:
            self.oai_server.shutdown()
            stop_thread(self.oai_thread)
