from dataclasses import dataclass

import torch
import hashlib
from typing import Optional, Tuple
from language_pipes.util.byte_helper import ByteHelper
from language_pipes.llm_layer_collector.state_obj import LLmComputationState

from language_pipes.util.utils import tensor_to_bytes, bytes_to_tensor, maybeTo

@dataclass
class JobData:
    cache_position: torch.Tensor
    causal_mask: torch.Tensor
    sliding_causal_mask: Optional[torch.Tensor]
    position_ids: torch.Tensor
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]
    position_embeddings_local: Optional[Tuple[torch.Tensor, torch.Tensor]]
    position_embeddings_global: Optional[Tuple[torch.Tensor, torch.Tensor]]
    state: torch.Tensor

    def hash_state(self):
        return hashlib.sha256(self.to_bytes()).digest()

    def to_bytes(self) -> bytes:
        state_bytes = tensor_to_bytes(self.state)
        cache_position_bytes = tensor_to_bytes(self.cache_position)
        causal_mask_bytes = tensor_to_bytes(self.causal_mask)
        sliding_causal_mask_bytes = tensor_to_bytes(self.sliding_causal_mask)
        position_ids_bytes = tensor_to_bytes(self.position_ids)
        position_embeddings_bytes = (
            (tensor_to_bytes(self.position_embeddings[0]), tensor_to_bytes(self.position_embeddings[1]))
            if self.position_embeddings is not None else (b'', b'')
        )
        position_embeddings_local_bytes = (
            (tensor_to_bytes(self.position_embeddings_local[0]), tensor_to_bytes(self.position_embeddings_local[1]))
            if self.position_embeddings_local is not None else (b'', b'')
        )
        position_embeddings_global_bytes = (
            (tensor_to_bytes(self.position_embeddings_global[0]), tensor_to_bytes(self.position_embeddings_global[1]))
            if self.position_embeddings_global is not None else (b'', b'')
        )

        bts = ByteHelper()

        bts.write_bytes(state_bytes)
        bts.write_bytes(cache_position_bytes)
        bts.write_bytes(causal_mask_bytes)
        bts.write_bytes(sliding_causal_mask_bytes)
        bts.write_bytes(position_ids_bytes)
        bts.write_bytes(position_embeddings_bytes[0])
        bts.write_bytes(position_embeddings_bytes[1])
        bts.write_bytes(position_embeddings_local_bytes[0])
        bts.write_bytes(position_embeddings_local_bytes[1])
        bts.write_bytes(position_embeddings_global_bytes[0])
        bts.write_bytes(position_embeddings_global_bytes[1])

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> Optional['JobData']:
        bts = ByteHelper(data)
        job_data = JobData(
            state = bytes_to_tensor(bts.read_bytes()),
            cache_position = bytes_to_tensor(bts.read_bytes()),
            causal_mask = bytes_to_tensor(bts.read_bytes()),
            sliding_causal_mask = bytes_to_tensor(bts.read_bytes()),
            position_ids = bytes_to_tensor(bts.read_bytes()),
            position_embeddings = (bytes_to_tensor(bts.read_bytes()), bytes_to_tensor(bts.read_bytes())),
            position_embeddings_local = (bytes_to_tensor(bts.read_bytes()), bytes_to_tensor(bts.read_bytes())),
            position_embeddings_global = (bytes_to_tensor(bts.read_bytes()), bytes_to_tensor(bts.read_bytes()))
        )

        return job_data

    @staticmethod
    def validate_state(data: bytes, state_hash: bytes) -> bool:
        current_hash = hashlib.sha256(data).digest()
        return current_hash == state_hash

def move_position_embeddings(t: Optional[Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if t is None:
        return None
    if str(t[0].device) == device:
        return (t[0].detach(), t[1].detach())
    return (
        t[0].detach().to(device),
        t[1].detach().to(device)
    )

def computationStateToJobData(data: LLmComputationState) -> JobData:
    return JobData(
        state=data.state.to('cpu'),
        position_ids=data.position_ids.to('cpu'),
        position_embeddings=move_position_embeddings(data.position_embeddings, torch.device('cpu')),
        position_embeddings_local=move_position_embeddings(data.position_embeddings_local, torch.device('cpu')),
        position_embeddings_global=move_position_embeddings(data.position_embeddings_global, torch.device('cpu')),
        cache_position=data.cache_position.to('cpu'),
        causal_mask=maybeTo(data.causal_mask["full_attention"], torch.device('cpu')), # type: ignore
        sliding_causal_mask=maybeTo(data.causal_mask["sliding_attention"], torch.device('cpu'))
    )

def jobDataToComputationState(data: JobData, device: torch.device) -> LLmComputationState:
    return LLmComputationState(
        state=data.state.to(device),
        position_ids=data.position_ids.to(device),
        position_embeddings=move_position_embeddings(data.position_embeddings, device),
        position_embeddings_local=move_position_embeddings(data.position_embeddings_local, device),
        position_embeddings_global=move_position_embeddings(data.position_embeddings_global, device),
        cache_position=data.cache_position.to(device),
        causal_mask={
            "full_attention": maybeTo(data.causal_mask, device),
            "sliding_attention": maybeTo(data.sliding_causal_mask, device)
        }
    )

def detachCompState(state: LLmComputationState) -> LLmComputationState:
    state.state = state.state.detach()
    state.position_ids = state.position_ids.detach()
    if state.position_embeddings is not None:
        state.position_embeddings = (state.position_embeddings[0].detach(), state.position_embeddings[1].detach())
    if state.position_embeddings_local is not None:
        state.position_embeddings_local = (state.position_embeddings_local[0].detach(), state.position_embeddings_local[1].detach())
    if state.position_embeddings_global is not None:
        state.position_embeddings_global = (state.position_embeddings_global[0].detach(), state.position_embeddings_global[1].detach())
    
    state.cache_position = state.cache_position.detach()
    state.causal_mask = {
        "full_attention": state.causal_mask["full_attention"].detach() if state.causal_mask["full_attention"] is not None else None,
        "sliding_attention": state.causal_mask["sliding_attention"].detach() if state.causal_mask["sliding_attention"] is not None else None
    }
    return state
