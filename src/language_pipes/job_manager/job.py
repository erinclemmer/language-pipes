import json
from typing import List, Optional
from uuid import uuid4

import torch

from distributed_state_network.objects.signed_packet import SignedPacket
from distributed_state_network.util.byte_helper import ByteHelper

from language_pipes.job_manager.job_data import JobData
from language_pipes.job_manager.enums import ComputeStep, JobStatus
from language_pipes.util.chat import ChatMessage
from language_pipes.util import tensor_to_bytes, bytes_to_tensor, bytes_to_int

class Job(SignedPacket):
    router_id: str
    from_router_id: str
    stop_token: int = 128009
    input_ids: List[int]
    prompt_tokens: int = 0
    tokens: int
    job_id: str
    pipe_id: str
    model_id: str
    current_layer: int
    current_step: ComputeStep
    status: JobStatus
    current_token: int = 0
    data: Optional[JobData]
    messages: List[ChatMessage]
    result: Optional[str]

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
            data: Optional[JobData] = None
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
        self.data = data if data is not None else JobData()

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

    def set_output(self, token: int):
        if self.current_step != ComputeStep.HEAD:
            raise Exception('Invalid step for head')
        self.input_ids.append(token)
        self.next_step()

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
            output = self.input_ids[-1]
            if output == self.stop_token:
                self.tokens = self.current_token
                self.status = JobStatus.COMPLETED
                return
            self.current_step = ComputeStep.EMBED
            if self.current_token == self.tokens:
                self.status = JobStatus.COMPLETED
        else:
            self.status = JobStatus.COMPLETED

    def to_bytes(self, include_signature: bool = True) -> bytes:
        bts = ByteHelper()
    
        bts.write_string(self.router_id)
        bts.write_string(self.from_router_id)
        bts.write_string(self.model_id)
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)

        if include_signature:
            bts.write_bytes(self.ecdsa_signature)

        bts.write_int(self.current_layer)
        bts.write_int(self.current_token)
        bts.write_int(self.tokens)
        bts.write_int(self.current_step.value)
        bts.write_int(self.status.value)
        bts.write_int(self.prompt_tokens if self.prompt_tokens is not None else 0)

        if self.input_id_tensor() is not None:
            input_ids_bytes = tensor_to_bytes(self.input_id_tensor())
            bts.write_bytes(input_ids_bytes)
        else:
            bts.write_int(0)

        messages_data = json.dumps([m.to_json() for m in self.messages]).encode('utf-8')
        bts.write_bytes(messages_data)

        if self.result is not None:
            bts.write_string(self.result)
        else:
            bts.write_int(0)

        data_bytes = self.data.to_bytes() if self.data is not None else b''
        bts.write_bytes(data_bytes)
        return bts.get_bytes()
    
    @staticmethod
    def from_bytes(data: bytes) -> 'Job':
        bts = ByteHelper(data)

        router_id = bts.read_string()
        from_router_id = bts.read_string()
        model_id = bts.read_string()
        job_id = bts.read_string()
        pipe_id = bts.read_string()
        ecdsa_signature = bts.read_bytes()
        current_layer = bts.read_int()
        current_token = bts.read_int()
        tokens = bts.read_int()
        current_step = ComputeStep(bts.read_int())
        status = JobStatus(bts.read_int())
        prompt_tokens = bts.read_int()
        
        input_ids = None
        input_ids_length = bytes_to_int(bts.bts.read(4))
        if input_ids_length > 0:
            input_ids = bytes_to_tensor(bts.bts.read(input_ids_length)).tolist()

        messages = [ChatMessage.from_dict(m) for m in json.loads(bts.read_string())]

        result = None
        num_result_bytes = bytes_to_int(bts.bts.read(4))
        if num_result_bytes > 0:
            result = bts.bts.read(num_result_bytes).decode('utf-8')
        data = JobData.from_bytes(bts.read_bytes())
    
        return Job(
            router_id=router_id,
            from_router_id=from_router_id,
            pipe_id=pipe_id,
            model_id=model_id,
            ecdsa_signature=ecdsa_signature,
            tokens=tokens,
            messages=messages,
            current_layer=current_layer,
            job_id=job_id,
            current_step=current_step,
            status=status,
            result=result,
            current_token=current_token,
            input_ids=input_ids,
            prompt_tokens=prompt_tokens,
            data=data
        )
