import os
import subprocess
import logging
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Callable
from llm_layer_collector.compute import compute_embedding, compute_head
from transformers.models.auto.tokenization_auto import AutoTokenizer
from llm_layer_collector.auto.auto_rms import AutoRMSNorm
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

import torch
from torch import tensor

from language_pipes.util.meta import MetaModel
from language_pipes.job_manager.job import ComputeStep, Job
from language_pipes.job_manager.job_data import computationStateToJobData, jobDataToComputationState
from language_pipes.llm_model.computed import ComputedData
from llm_layer_collector import LlmLayerCollector

class LlmModel:
    model_id: str
    computed: ComputedData
    process_id: str
    pipe_id: str
    collector: LlmLayerCollector

    router_id: str
    device: str
    virtual: bool

    input_embedding: Optional[torch.nn.Embedding] | bool
    layers: List[AutoDecoderLayer]
    norm: Optional[AutoRMSNorm] | bool
    head: Optional[torch.nn.Linear] | bool
    tokenizer: Callable

    start_layer: int
    end_layer: int
    loaded: bool
    num_hidden_layers: int

    def __init__(
            self,
            model_id: str,
            router_id: str,
            pipe_id: str,
            device: str,
            process_id: Optional[str] = None
    ):
        self.model_id = model_id
        self.router_id = router_id
        self.pipe_id = pipe_id
        self.loaded = False
        self.virtual = False
        self.layers = []
        self.input_embedding = None
        self.norm = None
        self.head = None
        self.start_layer = -1
        self.end_layer = -1
        self.device = device
        model_dir = os.path.join('models', self.model_id)
        if not os.path.exists(model_dir):
            self.clone_model(model_id, model_dir)
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

        self.computed = ComputedData(f'models/{model_id}')
        self.tokenizer = lambda: AutoTokenizer.from_pretrained(os.path.join(model_dir, 'data'))
        self.logger = logging.getLogger("LM NET: " + self.router_id)

    def clone_model(self, model_id: str, model_dir: str):
        repo_url = f"https://huggingface.co/{model_id}"
        clone_dir = f"{model_dir}/data"

        if not os.path.exists(clone_dir):
            Path(clone_dir).mkdir(parents=True)
        subprocess.run(["git", "clone", repo_url, clone_dir])
        subprocess.run(["git", "lfs", "install"], cwd=clone_dir, check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=clone_dir, check=True)

    def load(self):
        if self.end_layer > self.num_hidden_layers:
            self.end_layer = self.num_hidden_layers - 1

        if self.input_embedding:
            self.input_embedding = self.collector.load_input_embedding(self.device)
        
        self.norm = self.collector.load_norm(self.device)
        
        if self.head:
            self.head = self.collector.load_head(self.device)
        
        if self.start_layer == -1 or self.end_layer == -1:
            self.layers = []
        else:
            self.layers = self.collector.load_layer_set(self.start_layer, self.end_layer, self.device)
        self.loaded = True
        self.virtual = False

    def print(self):
        self.logger.info(f'''
#################################
Loaded Model: {self.model_id}
Pipe ID: {self.pipe_id}
Router: {self.router_id}
Process: {self.process_id}
Embed: {self.input_embedding is not None}
Head: {self.head is not None}
Start Layer: {self.start_layer}
End Layer: {self.end_layer}
Device: {self.device}
#################################
''')

    def process_job(self, job: Job):
        self.logger.info(f'Processing job state {job.current_step.name + (", layer " + str(job.current_layer)) if job.current_step.value == 2 else ""}')
        if job.current_step == ComputeStep.TOKENIZE:
            self.tokenize(job)
        elif job.current_step == ComputeStep.EMBED:
            self.compute_embed(job)
        elif job.current_step == ComputeStep.LAYER:
            self.compute_layers(job)
        elif job.current_step == ComputeStep.NORM:
            self.compute_norm(job)
        elif job.current_step == ComputeStep.HEAD:
            self.compute_head(job)

    def tokenize(self, job: Job):
        tokenizer: AutoTokenizer = self.tokenizer()
        prompt = tokenizer.apply_chat_template([m.to_json() for m in job.messages], tokenize=False, chat_template=tokenizer.chat_template, add_generation_prompt=True)
        input_tokens = [int(t) for t in tokenizer.encode(prompt, return_tensors='pt')[0].numpy()]
        job.input_ids = input_tokens
        job.prompt_tokens = len(input_tokens)
        job.next_step()

    def raise_exception(self, msg):
        self.logger.exception(msg)
        raise Exception(msg)

    def compute_layers(
        self, 
        job: Job,
    ):
        if job.data is None:
            self.raise_exception("cannot compute layers without job data")
        comp_state = jobDataToComputationState(job.data)
        for lyr in self.layers:
            comp_state.state = lyr(comp_state)
        job.set_layer(comp_state.state, self.end_layer + 1)
        if job.current_layer == self.num_hidden_layers:
            job.next_step()
    
    def compute_embed(self, job: Job):
        if job.current_step != ComputeStep.EMBED:
            self.raise_exception('Invalid step for embedding')
        if self.input_embedding is None:
            self.raise_exception("Input Embedding must be loaded before computation")
        comp_state = compute_embedding(self.input_embedding, tensor([job.input_ids]).to(self.device), self.collector.config)
        job.data = computationStateToJobData(comp_state)
        job.next_step()

    def compute_norm(self, job: Job):
        if job.data is None or job.data.state is None:
            self.raise_exception("Cannot compute norm without job data")
        norm = self.norm(job.data.state.to(self.device))
        job.set_norm(norm)
        
    def compute_head(self, job: Job):
        if self.head is None:
            self.raise_exception("Head must be loaded before computation")
        if job.data is None or job.data.state is None:
            self.raise_exception("Cannot compute head without job data")
        head = int(compute_head(self.head, job.data.state.to(self.device))[0][0])
        job.set_output(head)
    
    def to_meta(self) -> MetaModel:
        return MetaModel(
            process_id=self.process_id,
            has_embedding=self.input_embedding is not None and self.input_embedding != False,
            has_head=self.head is not None and self.head != False,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            router_id=self.router_id,
            pipe_id=self.pipe_id,
            model_id=self.model_id,
            loaded=self.loaded,
            num_layers=self.num_hidden_layers,
            computed=ComputedData.to_meta(self.computed)
        )

    def cleanup_tensors(self):
        del self.input_embedding
        del self.norm
        del self.head
        torch.cuda.empty_cache()
        del self.layers
        torch.cuda.empty_cache()

    @staticmethod
    def from_meta(meta: MetaModel) -> 'LlmModel':
        model = LlmModel(
            model_id=meta.model_id,
            router_id=meta.router_id,
            pipe_id=meta.pipe_id,
            device='cpu',
            process_id=meta.process_id
        )
        model.loaded = meta.loaded
        model.input_embedding = None if not meta.has_embedding else True
        model.head = None if not meta.has_head else True
        model.start_layer = meta.start_layer
        model.end_layer = meta.end_layer
        model.computed = ComputedData.from_meta(meta.computed)
        model.virtual = True

        return model
    
    @staticmethod
    def from_id(model_id: str, router_id: str, pipe_id: str, device: str) -> 'LlmModel':
        model = LlmModel(model_id, router_id, pipe_id, device)
        model.computed = ComputedData(f'models/{model_id}')
        return model
