import os
import gc
import ctypes
import logging
from pathlib import Path
from time import time, sleep
from uuid import uuid4
from threading import Thread
from typing import List, Optional, Callable, Dict
from transformers.cache_utils import DynamicCache

from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

import torch

from language_pipes.util import clone_model
from language_pipes.util.meta import MetaModel
from language_pipes.job_manager.job import Job
from language_pipes.job_manager.job_data import jobDataToComputationState, detachCompState
from language_pipes.llm_model.computed import ComputedData
from language_pipes.job_manager.layer_job import LayerJob, LayerTime
from llm_layer_collector import LlmLayerCollector

def compute_layers(job_data, device, layers, cache):
    comp_state = jobDataToComputationState(job_data, device)
    comp_state = detachCompState(comp_state)
    
    with torch.inference_mode():
        for lyr in layers:
            comp_state.state = lyr(comp_state, cache).detach()
    
    return comp_state.state.detach()

class LlmModel:
    model_id: str
    computed: ComputedData
    process_id: str
    pipe_id: str
    collector: LlmLayerCollector

    node_id: str
    device: str
    virtual: bool
    app_dir: str

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
            app_dir: str,
            process_id: Optional[str] = None
    ):
        self.model_id = model_id
        self.node_id = node_id
        self.pipe_id = pipe_id
        self.loaded = False
        self.virtual = False
        self.layers = []
        self.start_layer = -1
        self.end_layer = -1
        self.device = device
        self.app_dir = app_dir
        model_dir = str(Path(app_dir) / 'models' / self.model_id)
        if not os.path.exists(model_dir):
            clone_model(model_id, model_dir)
        self.collector = LlmLayerCollector(
                model_dir=os.path.join(model_dir, 'data'),
                cache_file=os.path.join(model_dir, 'cache.json'),
                device=device,
                dtype=torch.float16 
        )
        self.num_hidden_layers = self.collector.config.num_hidden_layers
        if process_id is None:
            self.process_id = str(uuid4())
        else:
            self.process_id = process_id

        self.computed = ComputedData(model_dir)
        self.logger = logging.getLogger("LM NET: " + self.node_id)
            
    def load(self):
        if self.end_layer > self.num_hidden_layers:
            self.end_layer = self.num_hidden_layers - 1

        if self.start_layer == -1 or self.end_layer == -1:
            self.layers = []
        else:
            self.layers = self.collector.load_layer_set(self.start_layer, self.end_layer, self.device)
        self.loaded = True
        self.virtual = False

    def print(self):
        self.logger.info(f'''
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

    def process_job(self, job: LayerJob, cache: DynamicCache):
        self.logger.info(f'Processing job layer {job.current_layer}')
        lt = LayerTime(
            node_id=self.node_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer
        )
        self.compute_layers(job, cache)
        job.data_hash = job.data.hash_state()
        lt.send_time = time()
        job.times.append(lt)

    def raise_exception(self, msg):
        self.logger.exception(msg)
        raise Exception(msg)

    def compute_layers(
        self, 
        job: Job,
        cache: DynamicCache
    ):
        if job.data is None:
            self.raise_exception("cannot compute layers without job data")
        
        job.set_layer(compute_layers(
            job.data,
            self.device, 
            self.layers, 
            cache,
        ), self.end_layer + 1)

        if job.current_layer == self.num_hidden_layers:
            job.done = True
    
    def to_meta(self) -> MetaModel:
        return MetaModel(
            process_id=self.process_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            router_id=self.node_id,
            pipe_id=self.pipe_id,
            model_id=self.model_id,
            loaded=self.loaded,
            num_layers=self.num_hidden_layers,
            computed=ComputedData.to_meta(self.computed)
        )

    def cleanup_tensors(self):
        torch.cuda.empty_cache()
        del self.layers
        torch.cuda.empty_cache()

    @staticmethod
    def from_meta(meta: MetaModel, app_dir: str) -> 'LlmModel':
        model = LlmModel(
            model_id=meta.model_id,
            node_id=meta.router_id,
            pipe_id=meta.pipe_id,
            device='cpu',
            app_dir=app_dir,
            process_id=meta.process_id
        )
        model.loaded = meta.loaded
        model.start_layer = meta.start_layer
        model.end_layer = meta.end_layer
        model.computed = ComputedData.from_meta(meta.computed)
        model.virtual = True

        return model
    
    @staticmethod
    def from_id(app_dir: str, model_id: str, node_id: str, pipe_id: str, device: str) -> 'LlmModel':
        model = LlmModel(
            model_id=model_id, 
            node_id=node_id, 
            pipe_id=pipe_id, 
            device=device, 
            app_dir=app_dir
        )

        model_dir = str(Path(app_dir) / 'models' / model_id)
        model.computed = ComputedData(model_dir)
        return model
