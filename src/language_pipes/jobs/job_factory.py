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

class JobFactory:
    config: LpConfig
    job_tracker: JobTracker
    router_pipes: RouterPipes

    get_layer_models: Callable[[], List[LlmModel]]

    def __init__(
        self, 
        logger,
        config: LpConfig,
        job_tracker: JobTracker,
        get_pipe_by_model_id: Callable[[str], Optional[Pipe]],
    ):
        self.config = config
        self.logger = logger
        self.job_tracker = job_tracker
        self.get_pipe_by_model_id = get_pipe_by_model_id

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
        pipe = self.get_pipe_by_model_id(model_id)
        if pipe is None:
            resolve('No pipe available')
            raise_exception(self.logger, f"Could not find pipe for model {model_id}")
            return

        job = Job(
            origin_node_id=self.config.node_id,
            messages=messages, 
            pipe_id=pipe.pipe_id, 
            model_id=pipe.model_id,
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            resolve=resolve,
            update=update,
            complete=self.job_tracker.complete_job
        )
        
        if self.config.print_job_data:
            job.print_job(self.logger)
        
        network_job = job.to_layer_job()
        pipe.send_job(network_job, self.config.node_id)
        self.job_tracker.jobs_pending.append(job)

        if start is not None:
            start(job)

        return job
