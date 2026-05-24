from dataclasses import dataclass

import torch
import hashlib
from typing import Dict, Optional, Tuple
from language_pipes.util.byte_helper import ByteHelper
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

from language_pipes.util.utils import tensor_to_bytes, bytes_to_tensor

def write_tensor_dict(d: Dict[str, Optional[torch.Tensor]]) -> bytes:
    bts = ByteHelper()
    bts.write_int(len(list(d.keys())))
    for key in d.keys():
        bts.write_string(key)
        if d[key] is None:
            bts.write_int(0)
        else:
            bts.write_int(1)
            bts.write_bytes(tensor_to_bytes(d[key]))

    return bts.get_bytes()


def read_tensor_dict(raw_data: bytes) -> Dict[str, Optional[torch.Tensor]]:
    bts = ByteHelper(raw_data)
    data = { }
    num_keys = bts.read_int()
    current_key = 0
    while current_key < num_keys:
        current_key += 1
        key = bts.read_string()
        if bts.read_int() == 1:
            data[key] = bytes_to_tensor(bts.read_bytes())
        else:
            data[key] = None 

    return data

@dataclass
class JobData:
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    causal_mask: Dict[str, Optional[torch.Tensor]]
    position_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    state: torch.Tensor

    def hash_state(self):
        return hashlib.sha256(self.to_bytes()).digest()

    def to_bytes(self) -> bytes:
        state_bytes = tensor_to_bytes(self.state)
        cache_position_bytes = tensor_to_bytes(self.cache_position)
        position_ids_bytes = tensor_to_bytes(self.position_ids)
        
        bts = ByteHelper()

        bts.write_bytes(state_bytes)
        bts.write_bytes(position_ids_bytes)
        bts.write_bytes(cache_position_bytes)
        bts.write_bytes(write_tensor_dict(self.causal_mask))
        bts.write_int(len(self.position_embeddings.keys()))
        for key in self.position_embeddings.keys():
            bts.write_string(key)
            bts.write_bytes(tensor_to_bytes(self.position_embeddings[key][0]))
            bts.write_bytes(tensor_to_bytes(self.position_embeddings[key][1]))

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> Optional['JobData']:
        bts = ByteHelper(data)
        state = bytes_to_tensor(bts.read_bytes())
        position_ids = bytes_to_tensor(bts.read_bytes())
        cache_position = bytes_to_tensor(bts.read_bytes())
        causal_mask = read_tensor_dict(bts.read_bytes())
        num_keys = bts.read_int()
        position_embeddings = { }
        current_key = 0
        while current_key < num_keys:
            key = bts.read_string()
            t1 = bytes_to_tensor(bts.read_bytes())
            t2 = bytes_to_tensor(bts.read_bytes())
            position_embeddings[key] = (t1, t2)
            current_key += 1
        
        job_data = JobData(
            state = state,
            position_ids = position_ids,
            cache_position = cache_position,
            causal_mask = causal_mask,
            position_embeddings = position_embeddings
        )

        return job_data

    @staticmethod
    def validate_state(data: bytes, state_hash: bytes) -> bool:
        current_hash = hashlib.sha256(data).digest()
        return current_hash == state_hash

def move_position_embeddings(t: Dict[str, Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    for key in t.keys():
        t[key] = (t[key][0].to(device), t[key][1].to(device))
    
    return t

def move_causal_mask(t: Dict[str, Optional[torch.Tensor]], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    for key in t.keys():
        if t[key] is not None:        
            t[key] = t[key].to(device) # type: ignore
    
    return t

def computationStateToJobData(data: LLmComputationState) -> JobData:
    return JobData(
        state=data.state.to('cpu'),
        position_ids=data.position_ids.to('cpu'),
        cache_position=data.cache_position.to('cpu'),
        causal_mask=move_causal_mask(data.causal_mask, torch.device('cpu')),
        position_embeddings=move_position_embeddings(data.position_embeddings, torch.device('cpu')),
    )

def jobDataToComputationState(data: JobData, device: torch.device) -> LLmComputationState:
    return LLmComputationState(
        state=data.state.to(device),
        position_ids=data.position_ids.to(device),
        cache_position=data.cache_position.to(device),
        causal_mask=move_causal_mask(data.causal_mask, device),
        position_embeddings=move_position_embeddings(data.position_embeddings, device),
    )

def detachCompState(state: LLmComputationState) -> LLmComputationState:
    state.state = state.state.detach()
    state.position_ids = state.position_ids.detach()
    state.cache_position = state.cache_position.detach()
    for key in state.causal_mask.keys():
        if state.causal_mask[key] is not None:
            state.causal_mask[key] = state.causal_mask[key].detach() # type: ignore
    
    for key in state.position_embeddings.keys():
        state.position_embeddings[key] = (state.position_embeddings[key][0].detach(), state.position_embeddings[key][1].detach())

    return state
