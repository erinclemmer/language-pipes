import torch
from typing import List
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job_data import JobData
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.jobs.job_data import jobDataToComputationState, detachCompState
from language_pipes.llm_layer_collector.auto.static_auto_model import StaticAutoModel

def compute_layers(start_layer: int, job_data: JobData, device: str, layers: List[AutoDecoderLayer], cache: DynamicCache):
    comp_state = jobDataToComputationState(job_data, device)
    comp_state = detachCompState(comp_state)

    first_layer_idx: int = layers[0].cls.self_attn.layer_idx # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
    start_layer -= first_layer_idx
    
    with torch.inference_mode():
        for lyr in layers[start_layer:]:
            comp_state.state = StaticAutoModel.compute_layer(lyr, comp_state, cache).detach()
    
    return comp_state.state.detach()
