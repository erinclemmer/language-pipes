import torch
from typing import Optional, Tuple
from distributed_state_network.util.byte_helper import ByteHelper
from llm_layer_collector.compute import LLmComputationState

from language_pipes.util import tensor_to_bytes, bytes_to_tensor

class JobData:
    cache_position: Optional[torch.Tensor] = None
    causal_mask: Optional[torch.Tensor] = None
    sliding_causal_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    position_embeddings: Optional[torch.Tensor] = None
    position_embeddings_local: Optional[torch.Tensor] = None
    position_embeddings_global: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None

    def to_bytes(self) -> bytes:
        state_bytes = tensor_to_bytes(self.state) if self.state is not None else b''
        cache_position_bytes = tensor_to_bytes(self.cache_position) if self.cache_position is not None else b''
        causal_mask_bytes = tensor_to_bytes(self.causal_mask) if self.causal_mask is not None else b''
        sliding_causal_mask_bytes = tensor_to_bytes(self.sliding_causal_mask) if self.sliding_causal_mask is not None else b''
        position_ids_bytes = tensor_to_bytes(self.position_ids) if self.position_ids is not None else b''
        position_embeddings_bytes = tensor_to_bytes(
            self.position_embeddings) if self.position_embeddings is not None else b''
        position_embeddings_local_bytes = tensor_to_bytes(
            self.position_embeddings_local) if self.position_embeddings_local is not None else b''
        position_embeddings_global_bytes = tensor_to_bytes(
            self.position_embeddings_global) if self.position_embeddings_global is not None else b''

        bts = ByteHelper()

        bts.write_bytes(state_bytes)
        bts.write_bytes(cache_position_bytes)
        bts.write_bytes(causal_mask_bytes)
        bts.write_bytes(sliding_causal_mask_bytes)
        bts.write_bytes(position_ids_bytes)
        bts.write_bytes(position_embeddings_bytes)
        bts.write_bytes(position_embeddings_local_bytes)
        bts.write_bytes(position_embeddings_global_bytes)

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> Optional['JobData']:
        job_data = JobData()
        bts = ByteHelper(data)
        job_data.state = bytes_to_tensor(bts.read_bytes())
        job_data.cache_position = bytes_to_tensor(bts.read_bytes())
        job_data.causal_mask = bytes_to_tensor(bts.read_bytes())
        job_data.sliding_causal_mask = bytes_to_tensor(bts.read_bytes())
        job_data.position_ids = bytes_to_tensor(bts.read_bytes())
        job_data.position_embeddings = bytes_to_tensor(bts.read_bytes())
        job_data.position_embeddings_local = bytes_to_tensor(bts.read_bytes())
        job_data.position_embeddings_global = bytes_to_tensor(bts.read_bytes())
    
        return job_data

def move_position_embeddings(t: Optional[Tuple[torch.Tensor, torch.Tensor]], device: str):
    if t is None:
        return None
    if t[0].device == device:
        return t
    return [
        t[0].to(device),
        t[1].to(device)
    ]

def computationStateToJobData(data: LLmComputationState) -> JobData:
    job_data = JobData()
    job_data.state = maybeTo(data.state, 'cpu')
    job_data.position_ids = maybeTo(data.position_ids, 'cpu')
    job_data.position_embeddings = move_position_embeddings(data.position_embeddings, 'cpu')
    job_data.position_embeddings_local = move_position_embeddings(data.position_embeddings_local, 'cpu')
    job_data.position_embeddings_global = move_position_embeddings(data.position_embeddings_global, 'cpu')
    job_data.cache_position = maybeTo(data.cache_position, 'cpu')
    job_data.causal_mask = maybeTo(data.causal_mask["full_attention"], 'cpu')
    job_data.sliding_causal_mask = data.causal_mask["sliding_attention"].to('cpu') if "sliding_attention" in data.causal_mask else None
    return job_data

def maybeTo(t: Optional[torch.Tensor], device: str) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device)

def jobDataToComputationState(data: JobData, device: str) -> LLmComputationState:
    state = LLmComputationState()
    state.state = maybeTo(data.state, device)
    state.position_ids = maybeTo(data.position_ids, device)
    
    if data.position_embeddings is not None:
        state.position_embeddings = move_position_embeddings(data.position_embeddings, device)
    if data.position_embeddings_local is not None:
        state.position_embeddings_local = move_position_embeddings(data.position_embeddings_local, device)
    if data.position_embeddings_global is not None:
        state.position_embeddings_global = move_position_embeddings(data.position_embeddings_global, device)
    
    state.cache_position = maybeTo(data.cache_position, device)
    state.causal_mask = {
        "full_attention": maybeTo(data.causal_mask, device),
        "sliding_attention": maybeTo(data.sliding_causal_mask, device)
    }
    return state
