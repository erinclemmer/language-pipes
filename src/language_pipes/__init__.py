import logging
from threading import Thread
from distributed_state_network import DSNodeServer

from language_pipes.job_manager import JobManager
from language_pipes.job_receiver import JobReceiver
from language_pipes.handlers.oai import OAIHttpServer

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
        self.job_manager = None
        self.router = DSNodeServer.start(self.config.router, self.print_pipes, self.print_pipes)
        self.job_manager = JobManager(self.router.node, self.config.processor)
        self.job_receiver = JobReceiver(self.config.processor, self.router.node, self.job_manager.get_pipe, self.job_manager.restart_job)
        if self.config.oai_port is not None:
            self.start_oai()

    def print_pipes(self):
        if self.job_manager is None:
            return
        for p in self.job_manager.router_pipes.network_pipes():
            p.print(self.job_manager.logger)

    def start_oai(self):
        self.oai_server = OAIHttpServer(self.config.oai_port, self.job_manager.complete)
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
        self.job_manager.stop()
        self.job_receiver.stop()
        self.router.stop()
        if self.config.oai_port is not None:
            self.oai_server.shutdown()
            stop_thread(self.oai_thread)
