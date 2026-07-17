import torch
from typing import List
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job_data import JobData
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.jobs.job_data import jobDataToComputationState, detachCompState
from llm_layer_collector.auto.static_auto_model import StaticAutoModel

def compute_layers(start_layer: int, job_data: JobData, device: torch.device, config: PretrainedConfig, layers: List[AutoDecoderLayer], cache: DynamicCache):
    # Cast incoming activations to the local layers' compute dtype so a node
    # running a different precision (e.g. 8-bit fp16) can feed a bf16 node and
    # vice versa. Use the first *floating-point* parameter: on an 8-bit node the
    # projection weights are quantized to int8, so next(parameters()) may be an
    # int8 tensor — casting activations to that would corrupt them.
    local_dtype = next((p.dtype for p in layers[0].cls.parameters() if p.is_floating_point()), None)
    comp_state = jobDataToComputationState(job_data, device, local_dtype)
    comp_state = detachCompState(comp_state)

    first_layer_idx: int = layers[0].cls.self_attn.layer_idx # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
    start_layer -= first_layer_idx
    
    with torch.inference_mode():
        for lyr in layers[start_layer:]:
            comp_state.state = StaticAutoModel.compute_layer(lyr, config, comp_state, cache).detach()

    return comp_state.state.detach(), comp_state.shared_kv_states
