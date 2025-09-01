import torch
from typing import Optional
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

def computationStateToJobData(data: LLmComputationState) -> JobData:
    job_data = JobData()
    job_data.state = data.state
    job_data.position_ids = data.position_ids
    job_data.position_embeddings = data.position_embeddings
    job_data.cache_position = data.cache_position
    job_data.causal_mask = data.causal_mask["full_attention"]
    job_data.sliding_causal_mask = data.causal_mask["sliding_attention"] if "sliding_attention" in data.causal_mask else None
    return job_data

def jobDataToComputationState(data: JobData) -> LLmComputationState:
    state = LLmComputationState()
    state.state = data.state
    state.position_ids = data.position_ids
    state.position_embeddings = data.position_embeddings
    state.cache_position = data.cache_position
    state.causal_mask = {
        "full_attention": data.causal_mask,
        "sliding_attention": data.sliding_causal_mask
    }
    return state
