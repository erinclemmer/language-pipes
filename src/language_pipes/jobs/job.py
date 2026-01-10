import json
from time import time
from typing import List, Optional
from uuid import uuid4

import torch

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.network_job import NetworkJob, LayerTime
from transformers.cache_utils import DynamicCache
from language_pipes.util.chunk_state import ChunkState
from promise import Promise
from typing import Callable

from language_pipes.util.enums import ComputeStep, JobStatus
from language_pipes.util.chat import ChatMessage

from language_pipes.util import tensor_to_bytes, bytes_to_tensor, bytes_to_int

class Job:
    # IDs
    job_id: str
    pipe_id: str
    model_id: str
    origin_node_id: str
    
    # Computed
    delta: str
    prompt_tokens: int = 0

    # State Info
    input_ids: List[int]
    compute_step: ComputeStep
    status: JobStatus
    current_token: int = 0
    current_layer: int = 0
    data: Optional[JobData]
    messages: List[ChatMessage]
    result: Optional[str]
    last_update: float
    times: List[List[LayerTime]]
    
    # API params
    top_k: int
    top_p: float
    min_p: float
    temperature: float
    presence_penalty: float
    max_completion_tokens: int
    
    # Prefill timing fields
    prefill_start_time: float
    chunk_start_time: float

    # Classes
    cache: DynamicCache
    chunking: ChunkState

    # Functions
    resolve: Promise | None
    update: Callable[["Job"], None] | None

    def __init__(
            self,
            origin_node_id: str,
            messages: List[ChatMessage],
            pipe_id: str,
            model_id: str,
            data: Optional[JobData] = None,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            max_completion_tokens: int = 1000,
            resolve: Promise | None = None,
            update: Callable[["Job"], None] | None = None
        ):
        self.pipe_id = pipe_id
        self.model_id = model_id
        self.job_id = str(uuid4())
        self.origin_node_id = origin_node_id
        
        self.status = JobStatus.IN_PROGRESS
        self.compute_step = ComputeStep.TOKENIZE

        self.delta = ''
        self.data = None
        self.result = None
        self.input_ids = []
        self.times = []
        self.prompt_tokens = 0
        self.current_token = 0
        self.messages = messages

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_completion_tokens = max_completion_tokens
        
        self.prefill_start_time = 0
        self.chunk_start_time = 0
        self.current_layer = 0

        self.cache = DynamicCache()
        self.chunking = ChunkState()
        self.resolve = resolve
        self.update = update
        self.last_update = time()

    def init_chunking(self, chunk_size: int):
        self.chunk_start_time = time()
        self.prefill_start_time = time()
        self.chunking.init(self.prompt_tokens, chunk_size)

    def set_layer(self, state: torch.Tensor, layer: int):
        if self.compute_step != ComputeStep.LAYER:
            raise Exception('Invalid step for layer')
        self.current_layer = layer
        if self.data is None: 
            return
        self.data.state = state

    def set_norm(self, state: torch.Tensor):
        if self.compute_step != ComputeStep.NORM:
            raise Exception('Invalid step for norm')
        if self.data is None:
            return
        self.data.state = state
        self.next_step()

    def set_output(self, token: int, eos_token: int):
        if self.compute_step != ComputeStep.HEAD:
            raise Exception('Invalid step for head')
        self.input_ids.append(token)
        self.next_step()
        if token == eos_token:
            self.status = JobStatus.COMPLETED

    def input_id_tensor(self):
        if self.input_ids is None:
            return None
        return torch.tensor(self.input_ids)

    def next_step(self):
        if self.compute_step == ComputeStep.TOKENIZE:
            self.compute_step = ComputeStep.EMBED
        elif self.compute_step == ComputeStep.EMBED:
            self.compute_step = ComputeStep.LAYER
        elif self.compute_step == ComputeStep.LAYER:
            self.compute_step = ComputeStep.NORM
        elif self.compute_step == ComputeStep.NORM:
            self.compute_step = ComputeStep.HEAD
        elif self.current_token < self.max_completion_tokens:
            self.current_token += 1
            self.compute_step = ComputeStep.EMBED
            if self.current_token == self.max_completion_tokens:
                self.status = JobStatus.COMPLETED
        else:
            self.status = JobStatus.COMPLETED

    def receive_layer_job(self, network_job: NetworkJob) -> bool:
        if network_job.job_id != self.job_id or network_job.pipe_id != self.pipe_id:
            return False
        if network_job.origin_node_id != self.origin_node_id:
            return False

        if network_job.compute_step == ComputeStep.HEAD and self.chunking.has_more():
            self.compute_step = ComputeStep.EMBED
            self.current_layer = 0
        else:
            self.compute_step = network_job.compute_step
            self.current_layer = network_job.current_layer
            
        self.data = network_job.data
        self.times.append(network_job.times)
        
        return True

    def to_layer_job(self) -> NetworkJob:
        data_hash = self.data.hash_state() if self.data is not None else b''
        return NetworkJob(
            job_id=self.job_id, 
            pipe_id=self.pipe_id, 
            origin_node_id=self.origin_node_id, 
            current_layer=self.current_layer, 
            data=self.data, 
            data_hash=data_hash, 
            compute_step=self.compute_step, 
            times=[]
        )

    def set_last_update(self):
        from time import time
        self.last_update = time()

    def print_job(self, logger):
        logger.info(f"""
=================================
Job ID: {self.job_id}
Pipe ID: {self.pipe_id}
Prompt Tokens: {self.prompt_tokens}
Current Token: {self.current_token}
Max Tokens: {self.max_completion_tokens}
Temperature: {self.temperature}
Top K: {self.top_k}
Top P: {self.top_p}
Min P: {self.min_p}
Pres Penalty: {self.presence_penalty}
=================================
""")
