import gc
import psutil
from threading import Thread
from typing import Optional, List

import torch

from language_pipes.util import stop_thread
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.oai_server import OAIHttpServer
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.tui.content_provider.model_provider import ModelProvider
from language_pipes.tui.content_provider.network_provider import NetworkProvider
from language_pipes.tui.content_provider.pipe_provider import PipeProvider


class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    pipe_manager: Optional[PipeManager]
    job_tracker: Optional[JobTracker]
    job_factory: Optional[JobFactory]
    job_receiver: Optional[JobReceiver]
    oai_server: Optional[OAIHttpServer]
    oai_thread: Optional[Thread]
    model_manager: ModelManager
    model_provider: ModelProvider
    network_provider: NetworkProvider
    pipe_provider: PipeProvider

    def __init__(self):
        self.router = None
        self.router_pipes = None
        self.pipe_manager = None
        self.job_tracker = None
        self.job_factory = None
        self.job_receiver = None
        self.model_manager = ModelManager()
        
        self.model_provider = ModelProvider(lambda: self.model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(lambda: self.router, self.set_router)
        self.pipe_provider = PipeProvider(lambda: self.pipe_manager)

    def set_router(self, router: Optional[DSNodeServer]):
        self.router = router
        if router is not None:
            self.router_pipes = RouterPipes(router)
            self.pipe_manager = PipeManager(self.model_manager, self.router_pipes)
        else:
            self.router_pipes = None
            self.pipe_manager = None

    def start_oai_server(self, port: int, oai_keys: List[str]):
        if self.router is None or self.pipe_manager is None:
            raise Exception("Cannot start oai server without pipe manager")
        
        self.job_tracker = JobTracker()
        self.job_factory = JobFactory(self.job_tracker, self.pipe_manager)
        self.job_receiver = JobReceiver(
            job_factory=self.job_factory,
            job_tracker=self.job_tracker,
            model_manager=self.model_manager,
            pipe_manager=self.pipe_manager,
            is_shutdown=self.router.is_shut_down
        )

        def get_models():
            if self.router_pipes is None:
                return
            available_models = self.router_pipes.get_models(0)
            return [m.model_id for m in self.model_manager.end_models if m.model_id in available_models]

        self.oai_server = OAIHttpServer(
            api_keys=oai_keys,
            port=port,
            get_models=get_models,
            complete=self.job_factory.start_job
        )
        self.oai_thread = Thread(target=self.oai_server.serve_forever, args=())
        self.oai_thread.start()

    def stop_oai_server(self):
        if self.oai_thread is None or self.oai_server is None:
            return
        
        self.job_tracker = None
        self.job_factory = None
        self.job_receiver = None
        self.oai_server.shutdown()
        stop_thread(self.oai_thread)

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_total_system_ram() -> float:
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_used_system_ram() -> float:
        return psutil.virtual_memory().used / (1024**3)
