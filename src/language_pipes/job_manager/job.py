import json
from typing import List, Optional
from uuid import uuid4

import torch

from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

from language_pipes.job_manager.job_data import JobData
from language_pipes.job_manager.enums import ComputeStep, JobStatus
from language_pipes.util.chat import ChatMessage
from language_pipes.job_manager.layer_job import LayerJob
from language_pipes.util import tensor_to_bytes, bytes_to_tensor, bytes_to_int

class Job(SignedPacket):
    router_id: str
    from_router_id: str
    input_ids: List[int]
    prompt_tokens: int = 0
    tokens: int
    job_id: str
    pipe_id: str
    model_id: str
    delta: str
    current_layer: int
    current_step: ComputeStep
    status: JobStatus
    current_token: int = 0
    data: Optional[JobData]
    messages: List[ChatMessage]
    result: Optional[str]
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    # Prefill timing fields
    prefill_start_time: float
    chunk_start_time: float

    def __init__(
            self,
            router_id: str,
            from_router_id: str,
            tokens: int,
            messages: List[ChatMessage],
            pipe_id: str,
            model_id: str,
            ecdsa_signature: Optional[bytes] = None,
            current_layer: int = 0,
            job_id: str = "",
            current_step: ComputeStep = ComputeStep.TOKENIZE,
            status: JobStatus = JobStatus.IN_PROGRESS,
            current_token: int = 0,
            result: Optional[str] = None,
            input_ids: List[int] = [],
            prompt_tokens: int = 0,
            data: Optional[JobData] = None,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0
        ):
        super().__init__(ecdsa_signature)
        self.router_id = router_id
        self.from_router_id = from_router_id
        self.model_id = model_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.prompt_tokens = prompt_tokens
        self.job_id = str(uuid4()) if job_id == "" else job_id
        self.pipe_id = pipe_id
        self.current_layer = current_layer
        self.current_step = current_step
        self.status = status
        self.current_token = current_token
        self.result = result
        self.messages = messages
        self.delta = ''
        self.data = data if data is not None else JobData()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.prefill_start_time = 0.0
        self.chunk_start_time = 0.0

    def set_layer(self, state: torch.Tensor, layer: int):
        if self.current_step != ComputeStep.LAYER:
            raise Exception('Invalid step for layer')
        self.current_layer = layer
        if self.data is None: 
            return
        self.data.state = state

    def set_norm(self, state: torch.Tensor):
        if self.current_step != ComputeStep.NORM:
            raise Exception('Invalid step for norm')
        if self.data is None:
            return
        self.data.state = state
        self.next_step()

    def set_output(self, token: int, eos_token: int):
        if self.current_step != ComputeStep.HEAD:
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
        if self.current_step == ComputeStep.TOKENIZE:
            self.current_step = ComputeStep.EMBED
        elif self.current_step == ComputeStep.EMBED:
            self.current_step = ComputeStep.LAYER
        elif self.current_step == ComputeStep.LAYER:
            self.current_step = ComputeStep.NORM
            self.current_layer = 0
        elif self.current_step == ComputeStep.NORM:
            self.current_step = ComputeStep.HEAD
        elif self.current_token < self.tokens:
            self.current_token += 1
            self.current_step = ComputeStep.EMBED
            if self.current_token == self.tokens:
                self.status = JobStatus.COMPLETED
        else:
            self.status = JobStatus.COMPLETED

    def to_layer_job(self) -> LayerJob:
        return LayerJob(self.job_id, self.pipe_id, self.router_id, self.current_layer, self.data, self.data.hash_state(), False, False, [])

    def print_job(self, logger):
        logger.info(f"""
=================================
Job ID: {self.job_id}
Pipe ID: {self.pipe_id}
Prompt Tokens: {self.prompt_tokens}
Current Token: {self.current_token}
Max Tokens: {self.tokens}
Temperature: {self.temperature}
Top K: {self.top_k}
Top P: {self.top_p}
Min P: {self.min_p}
Pres Penalty: {self.presence_penalty}
=================================
""")
