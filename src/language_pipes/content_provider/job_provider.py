import gc
from pathlib import Path
from threading import Thread
from time import sleep, time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from language_pipes.config import LpConfig
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.oai_server import OAIHttpServer
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.util.utils import stop_thread

@dataclass
class MetaJob:
    job_id: str
    pipe_id: str
    model_id: str
    origin_node_id: str
    current_token: int
    last_update: float

class JobProvider:
    oai_server: Optional[OAIHttpServer]
    oai_thread: Optional[Thread]
    job_tracker: Optional[JobTracker]
    job_factory: Optional[JobFactory]
    job_receiver: Optional[JobReceiver]
    get_router_pipes: Callable[[], Optional[RouterPipes]]
    get_model_manager: Callable[[], Optional[ModelManager]]
    get_pipe_manager: Callable[[], Optional[PipeManager]]

    def __init__(self, get_router_pipes: Callable, get_model_manager: Callable, get_pipe_manager: Callable):
        self.get_router_pipes = get_router_pipes
        self.get_model_manager = get_model_manager
        self.get_pipe_manager = get_pipe_manager
        self.job_tracker = None
        self.job_factory = None
        self.job_receiver = None
        self.oai_server = None
        self.oai_thread = None

    @staticmethod
    def get_oai_port(config_file: Path) -> int:
        cfg = LpConfig.from_file(config_file)
        return cfg.oai_port
    
    @staticmethod
    def set_oai_port(config_file: Path, port: int):
        cfg = LpConfig.from_file(config_file)
        cfg.oai_port = port
        cfg.save()
        
    @staticmethod
    def get_api_keys(config_file: Path) -> List[str]:
        cfg = LpConfig.from_file(config_file)
        return cfg.api_keys
        
    @staticmethod
    def set_api_keys(config_file: Path, keys: List[str]):
        cfg = LpConfig.from_file(config_file)
        cfg.api_keys = keys
        cfg.save()

    def start_oai_server(self, args: Tuple[int, List[str]]):
        port, oai_keys = args
        router_pipes = self.get_router_pipes()
        model_manager = self.get_model_manager()
        pipe_manager = self.get_pipe_manager()
        if router_pipes is None or pipe_manager is None or model_manager is None:
            return
        
        self.job_tracker = JobTracker()
        self.job_factory = JobFactory(self.job_tracker, pipe_manager)
        self.job_receiver = JobReceiver(
            job_factory=self.job_factory,
            job_tracker=self.job_tracker,
            model_manager=model_manager,
            pipe_manager=pipe_manager,
            is_shutdown=router_pipes.router.is_shut_down
        )

        router_pipes.router.set_receive_cb(self.job_receiver.receive_data)

        def get_models():
            if router_pipes is None:
                return
            available_models = router_pipes.get_models(0)
            return [m.model_id for m in model_manager.end_models if m.model_id in available_models]

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
        if self.job_tracker is not None:
            self.job_tracker.shutdown = True
            sleep(0.1)
            self.job_tracker = None
        self.job_factory = None
        if self.job_receiver is not None:
            self.job_receiver.shutdown = True
            sleep(0.1)
            self.job_receiver = None
        self.oai_server.shutdown()
        self.oai_server.server_close()
        stop_thread(self.oai_thread)

        self.oai_server = None
        self.oai_thread = None

        gc.collect()
        torch.cuda.empty_cache()

    def oai_server_running(self) -> bool:
        return self.oai_server is not None
    
    def get_oai_logs(self) -> List[Tuple[float, str]]:
        if self.oai_server is None:
            return []
        return self.oai_server.logs

    def get_active_jobs(self) -> List[MetaJob]:
        if self.job_tracker is None:
            return []
        
        meta_jobs = []
        for job in self.job_tracker.jobs_pending:
            meta_jobs.append(MetaJob(
                job_id=job.job_id,
                pipe_id=job.pipe_id,
                model_id=job.model_id,
                current_token=job.current_token,
                origin_node_id=job.origin_node_id,
                last_update=time() - job.last_update
            ))
        
        return meta_jobs