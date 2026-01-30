from time import time
from uuid import uuid4
from typing import List, Optional

import torch
from promise import Promise
from typing import Callable
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.timing_stats import TimingStats
from language_pipes.util.chunk_state import ChunkState

from language_pipes.util.enums import ComputeStep, JobStatus
from language_pipes.util.chat import ChatMessage

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
    timing_stats: TimingStats
    
    # API params
    top_k: int
    top_p: float
    min_p: float
    temperature: float
    presence_penalty: float
    max_completion_tokens: int

    # Classes
    cache: DynamicCache
    chunking: ChunkState

    # Functions
    resolve: Promise | None
    update: Optional[Callable[["Job"], None]]
    complete: Callable[[], None]

    def __init__(
            self,
            origin_node_id: str,
            messages: List[ChatMessage],
            pipe_id: str,
            model_id: str,
            prefill_chunk_size: int,
            data: Optional[JobData] = None,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            max_completion_tokens: int = 1000,
            resolve: Optional[Promise] = None,
            update: Optional[Callable[["Job"], None]] = None,
            complete: Optional[Callable[["Job"], None]] = None
        ):
        self.pipe_id = pipe_id
        self.model_id = model_id
        self.job_id = str(uuid4())
        self.origin_node_id = origin_node_id
        
        self.status = JobStatus.IN_PROGRESS
        self.compute_step = ComputeStep.TOKENIZE

        self.delta = ''
        self.data = data
        self.result = None
        self.input_ids = []
        self.timing_stats = TimingStats(self.job_id, prefill_chunk_size)
        self.prompt_tokens = 0
        self.current_token = 0
        self.messages = messages

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_completion_tokens = max_completion_tokens
        
        self.current_layer = 0

        self.cache = DynamicCache()
        self.chunking = ChunkState(self.job_id)
        self.resolve = resolve
        self.update = update

        if complete is not None:
            self.complete = lambda: complete(self)
        else:
            self.complete = self.pass_complete
        
        self.last_update = time()

    def pass_complete(self):
        pass

    def init_chunking(self, chunk_size: int):
        self.chunking.init(self.prompt_tokens, chunk_size)

    def set_layer(self, state: torch.Tensor, layer: int, num_hidden_layers: int):
        if self.compute_step != ComputeStep.LAYER:
            raise Exception('Invalid step for layer')
        self.current_layer = layer
        if self.data is None: 
            return
        self.data.state = state
        if self.current_layer == num_hidden_layers:
            self.compute_step = ComputeStep.EMBED if self.chunking.has_more() else ComputeStep.HEAD
            self.current_layer = 0

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

    def receive_network_job(self, network_job: NetworkJob) -> bool:
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
        self.timing_stats.receive_network_job(network_job.times)
        
        return True

    def send_update(self):
        self.last_update_time = time()
        if self.update is not None:
            return self.update(self)
        return True

    def to_network_job(self) -> NetworkJob:
        data_hash = self.data.hash_state() if self.data is not None else b''
        return NetworkJob(
            job_id=self.job_id, 
            pipe_id=self.pipe_id, 
            origin_node_id=self.origin_node_id, 
            current_layer=self.current_layer, 
            data=self.data, 
            data_hash=data_hash, 
            compute_step=self.compute_step, 
            times=list(self.timing_stats.current_times)
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
