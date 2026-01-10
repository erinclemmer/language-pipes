import gc
import ctypes
import random
import requests
from time import time, sleep
from typing import List, Optional, Tuple, Callable

import torch
from promise import Promise

from uuid import uuid4
from threading import Thread
from distributed_state_network import DSNode

from language_pipes.util import raise_exception
from language_pipes.util.enums import JobStatus
from language_pipes.util.chat import ChatMessage

from language_pipes.modeling.end_model import EndModel

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.pipes.pipe import Pipe

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_tracker import JobTracker

from language_pipes.config import LpConfig

from language_pipes.modeling.llm_model import LlmModel
from language_pipes.modeling.computed import validate_model

class JobManager:
    app_dir: str
    started: bool

    router: DSNode
    config: LpConfig
    job_tracker: JobTracker
    router_pipes: RouterPipes

    get_end_model: Callable[[str], Optional[EndModel]]

    def __init__(
        self, 
        app_dir: str, 
        router: DSNode, 
        config: LpConfig,
        router_pipes: RouterPipes,
        job_tracker: JobTracker,
        get_layer_models: Callable[[], List[LlmModel]],
        get_end_model: Callable[[str], Optional[EndModel]]
    ):
        self.started = False
        self.router = router
        self.config = config
        self.app_dir = app_dir
        self.logger = self.router.logger
        self.get_layer_models = get_layer_models
        self.get_end_model = get_end_model

        self.job_tracker = job_tracker
        self.router_pipes = router_pipes
        self.router.update_data("job_port", str(self.config.job_port))
        
        self.router_pipes.print_pipes()

        self.started = True
    
    def get_pipe(self, pipe_id: str) -> Optional[Pipe]:
        meta_pipe = self.router_pipes.network_pipe(pipe_id)
        if meta_pipe is None:
            return None
        return Pipe.from_meta(
            meta_pipe=meta_pipe,
            hosted_models=self.get_layer_models(),
            router=self.router,
            app_dir=self.app_dir,
            get_job_port=self.get_job_port,
            complete_job=self.job_tracker.complete_job,
            send_job_update=self.job_tracker.send_job_update
        )
    
    def get_job_port(self, node_id: str) -> Optional[int]:
        try:
            return int(self.router.read_data(node_id, 'job_port'))
        except Exception as e:
            self.logger.exception("Error getting job port: %s", e)
            return None

    def start_job(
        self, 
        model_id: str, 
        messages: List[ChatMessage], 
        max_completion_tokens: int, 
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        start: Optional[Callable] = None,
        update: Optional[Callable] = None,
        resolve: Optional[Promise] = None
    ) -> Optional[Job]:
        end_model = self.get_end_model(model_id)
        if end_model is None:
            if resolve is not None:
                resolve('NO_ENDS')
                return None
            raise_exception(self.logger, f"Could not find local end model for {model_id}")
            return

        network_pipe = self.router_pipes.get_job_pipe(model_id)
        if network_pipe is None:
            if resolve is not None:
                resolve('NO_PIPE')
                return None
            raise_exception(self.logger, f"Could not find pipe for {model_id}")
            return
        pipe_id = network_pipe.pipe_id

        pipe = self.get_pipe(pipe_id)
        if pipe is None:
            raise_exception(self.logger, f"Could not find pipe {pipe_id}")
            return

        job = Job(
            origin_node_id=self.router.config.node_id, 
            messages=messages, 
            pipe_id=pipe_id, 
            model_id=pipe.model_id,
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            resolve=resolve,
            update=update
        )
        
        # Tokenize first to get prompt length
        end_model.tokenize(job)
        
        # Init chunking
        self.logger.info(f"[Prefill] job={job.job_id[:8]} started")
        job.init_chunking(self.config.prefill_chunk_size)
        job.chunking.print_start(self.logger)
        
        if self.config.print_job_data:
            job.print_job(self.router.logger)
        
        network_job = job.to_layer_job()
        pipe.send_job(network_job, self.router.config.node_id)
        self.job_tracker.jobs_pending.append(job)

        if start is not None:
            start(job)

        return job
