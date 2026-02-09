import torch
from typing import List
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job_data import JobData
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.jobs.job_data import jobDataToComputationState, detachCompState
from language_pipes.llm_layer_collector.auto.static_auto_model import StaticAutoModel

def compute_layers(job_data: JobData, device: str, layers: List[AutoDecoderLayer], cache: DynamicCache):
    comp_state = jobDataToComputationState(job_data, device)
    comp_state = detachCompState(comp_state)
    
    with torch.inference_mode():
        for lyr in layers:
            comp_state.state = StaticAutoModel.compute_layer(lyr, comp_state, cache).detach()
    
    return comp_state.state.detach()
