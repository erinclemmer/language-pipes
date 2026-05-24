import gc
from pathlib import Path
from threading import Thread
from time import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from language_pipes.config import LpConfig
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.modeling.end_model import EndModel
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
    prompt_processed: float
    last_update: float

class JobProvider:
    oai_server: Optional[OAIHttpServer]
    oai_thread: Optional[Thread]
    config_file: Path
    get_job_tracker: Callable[[], Optional[JobTracker]]
    get_job_factory: Callable[[], Optional[JobFactory]]
    get_job_receiver: Callable[[], Optional[JobReceiver]]
    get_router_pipes: Callable[[], Optional[RouterPipes]]
    get_model_manager: Callable[[], Optional[ModelManager]]
    get_pipe_manager: Callable[[], Optional[PipeManager]]

    def __init__(
            self, 
            config_file: Path, 
            get_router_pipes: Callable, 
            get_model_manager: Callable, 
            get_pipe_manager: Callable,
            get_job_tracker: Callable,
            get_job_factory: Callable,
            get_job_receiver: Callable
        ):
        self.get_router_pipes = get_router_pipes
        self.get_model_manager = get_model_manager
        self.get_pipe_manager = get_pipe_manager
        self.get_job_tracker = get_job_tracker
        self.get_job_factory = get_job_factory
        self.get_job_receiver = get_job_receiver
        self.config_file = config_file
        self.oai_server = None
        self.oai_thread = None

    def get_oai_port(self) -> int:
        cfg = LpConfig.from_file(self.config_file)
        return cfg.oai_port
    
    def set_oai_port(self, port: int):
        cfg = LpConfig.from_file(self.config_file)
        cfg.oai_port = port
        cfg.save()
        
    def get_api_keys(self) -> List[str]:
        cfg = LpConfig.from_file(self.config_file)
        return cfg.api_keys
        
    def set_api_keys(self, keys: List[str]):
        cfg = LpConfig.from_file(self.config_file)
        cfg.api_keys = keys
        cfg.save()

    def start_oai_server(self, cfg: Optional[LpConfig] = None):
        router_pipes = self.get_router_pipes()
        model_manager = self.get_model_manager()
        job_factory = self.get_job_factory()
        
        if router_pipes is None or model_manager is None or job_factory is None:
            return
        
        def get_models():
            if router_pipes is None:
                return
            available_models = router_pipes.get_models(EndModel.get_num_local_layers())
            return [m.model_id for m in model_manager.end_models if m.model_id in available_models]

        if cfg is None:
            cfg = LpConfig.from_file(self.config_file)
        
        self.oai_server = OAIHttpServer(
            api_keys=cfg.api_keys,
            port=cfg.oai_port,
            get_models=get_models,
            complete=job_factory.start_job
        )
        self.oai_thread = Thread(target=self.oai_server.serve_forever, args=())
        self.oai_thread.start()

    def stop_oai_server(self):
        if self.oai_thread is None or self.oai_server is None:
            return
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
    
    def reset_oai_logs(self):
        if self.oai_server is None:
            return
        self.oai_server.logs = []

    def get_active_jobs(self) -> List[MetaJob]:
        job_tracker = self.get_job_tracker()
        if job_tracker is None:
            return []
        
        meta_jobs = []
        for job in job_tracker.jobs_pending:
            meta_jobs.append(MetaJob(
                job_id=job.job_id,
                pipe_id=job.pipe_id,
                model_id=job.model_id,
                current_token=job.current_token,
                origin_node_id=job.origin_node_id,
                prompt_processed=(job.chunking.current_chunk / job.chunking.total_chunks) if job.chunking.total_chunks > 0 else 1,
                last_update=time() - job.last_update
            ))
        
        return meta_jobs