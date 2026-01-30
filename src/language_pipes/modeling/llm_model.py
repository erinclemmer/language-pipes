import os
import logging
from pathlib import Path
from uuid import uuid4
from logging import Logger
from typing import List, Optional, Callable

import torch

from llm_layer_collector import LlmLayerCollector
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

from language_pipes.util import clone_model

from language_pipes.modeling.meta_model import MetaModel
from language_pipes.modeling.llm_meta_data import LlmMetadata

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import jobDataToComputationState, detachCompState

def compute_layers(job_data, device, layers, cache):
    comp_state = jobDataToComputationState(job_data, device)
    comp_state = detachCompState(comp_state)
    
    with torch.inference_mode():
        for lyr in layers:
            comp_state.state = lyr(comp_state, cache).detach()
    
    return comp_state.state.detach()

class LlmModel:
    model_id: str
    meta_data: LlmMetadata
    process_id: str
    pipe_id: str
    collector: LlmLayerCollector

    node_id: str
    device: str
    virtual: bool
    model_dir: str

    layers: List[AutoDecoderLayer]
    tokenizer: Callable

    start_layer: int
    end_layer: int
    loaded: bool
    num_hidden_layers: int

    def __init__(
            self,
            model_id: str,
            node_id: str,
            pipe_id: str,
            device: str,
            model_dir: str,
            process_id: Optional[str] = None,
            virtual: bool = False
    ):
        self.model_id = model_id
        self.node_id = node_id
        self.pipe_id = pipe_id
        self.loaded = False
        self.virtual = virtual
        self.layers = []
        self.start_layer = -1
        self.end_layer = -1
        self.device = device
        self.model_dir = model_dir

        if not virtual:
            model_path = str(Path(model_dir) / self.model_id)
            if not os.path.exists(model_path):
                clone_model(model_id, model_path)
            self.collector = LlmLayerCollector(
                    model_dir=os.path.join(model_path, 'data'),
                    cache_file=os.path.join(model_path, 'cache.json'),
                    device=device,
                    dtype=torch.float16 
            )
            self.num_hidden_layers = self.collector.config.num_hidden_layers
            self.meta_data = LlmMetadata(model_path)

        if process_id is None:
            self.process_id = str(uuid4())
        else:
            self.process_id = process_id
            
    def load(self):
        if self.end_layer > self.num_hidden_layers:
            self.end_layer = self.num_hidden_layers - 1

        if self.start_layer == -1 or self.end_layer == -1:
            self.layers = []
        else:
            self.layers = self.collector.load_layer_set(self.start_layer, self.end_layer, self.device)
        self.loaded = True
        self.virtual = False

    def print(self, logger: logging.Logger):
        logger.info(f'''
=================================
Loaded Model: {self.model_id}
Pipe ID: {self.pipe_id}
Node: {self.node_id}
Process: {self.process_id}
Start Layer: {self.start_layer}
End Layer: {self.end_layer}
Device: {self.device}
=================================
''')

    def process_job(self, job: Job, logger: Logger):
        job.timing_stats.add_layer_time(self.node_id, job.current_layer, self.end_layer)
        self.compute_layers(job)
        job.timing_stats.set_send_time(logger)

    def compute_layers(
        self, 
        job: Job
    ):
        if job.data is None:
            raise Exception("cannot compute layers without job data")
        
        job.set_layer(
            state=compute_layers(
                job.data,
                self.device, 
                self.layers, 
                job.cache,
            ), 
            layer=self.end_layer + 1, 
            num_hidden_layers=self.num_hidden_layers
        )
    
    def to_meta(self) -> MetaModel:
        return MetaModel(
            process_id=self.process_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            node_id=self.node_id,
            pipe_id=self.pipe_id,
            model_id=self.model_id,
            loaded=self.loaded,
            num_layers=self.num_hidden_layers,
            meta_data=self.meta_data
        )

    def cleanup_tensors(self):
        torch.cuda.empty_cache()
        del self.layers
        torch.cuda.empty_cache()

    @staticmethod
    def from_meta(meta: MetaModel, model_dir: str) -> 'LlmModel':
        model = LlmModel(
            model_id=meta.model_id,
            node_id=meta.node_id,
            pipe_id=meta.pipe_id,
            device='cpu',
            model_dir=model_dir,
            process_id=meta.process_id,
            virtual=True
        )
        model.loaded = meta.loaded
        model.start_layer = meta.start_layer
        model.end_layer = meta.end_layer
        model.meta_data = meta.meta_data
        model.virtual = True

        return model
    
    @staticmethod
    def from_id(model_dir: str, model_id: str, node_id: str, pipe_id: str, device: str) -> 'LlmModel':
        model = LlmModel(
            model_id=model_id, 
            node_id=node_id, 
            pipe_id=pipe_id, 
            device=device, 
            model_dir=model_dir
        )

        model_path = str(Path(model_dir) / model_id)
        model.meta_data = LlmMetadata(model_path)
        return model
