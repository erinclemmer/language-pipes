import os
import torch
from uuid import uuid4
from torch import tensor
from pathlib import Path
from logging import Logger

from transformers.models.auto.tokenization_auto import AutoTokenizer

from language_pipes.llm_layer_collector import LlmLayerCollector
from language_pipes.llm_layer_collector.auto.auto_rms import AutoRMSNorm
from language_pipes.llm_layer_collector.compute import compute_embedding, compute_head

from language_pipes.jobs.job import ComputeStep, Job
from language_pipes.jobs.job_data import computationStateToJobData

from language_pipes.modeling.llm_meta_data import LlmMetadata

from language_pipes.util import clone_model

class EndModel:
    model_id: str
    process_id: str
    device: str
    input_embedding: torch.nn.Embedding
    norm: AutoRMSNorm
    head: torch.nn.Linear
    collector: LlmLayerCollector

    def __init__(self, model_dir: str, model_id: str, device: str):
        self.model_id = model_id
        self.device = device

        self.process_id = str(uuid4())
        model_path = str(Path(model_dir) / self.model_id)
        if not os.path.exists(model_path):
            clone_model(model_id, model_path)
        self.meta_data = LlmMetadata(model_path)
        self.collector = LlmLayerCollector(
            model_dir=os.path.join(model_path, 'data'),
            cache_file=os.path.join(model_path, 'cache.json'),
            device=device,
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'data'))
    
    def size(self):
        return self.meta_data.embed_size + self.meta_data.head_size

    def load(self):
        self.input_embedding = self.collector.load_input_embedding(self.device)
        self.norm = self.collector.load_norm(self.device)
        self.head = self.collector.load_head(self.device)

    def tokenize(self, job: Job):
        prompt = self.tokenizer.apply_chat_template([m.to_json() for m in job.messages], tokenize=False, chat_template=self.tokenizer.chat_template, add_generation_prompt=True)
        input_tokens = [int(t) for t in self.tokenizer.encode(prompt, return_tensors='pt')[0].numpy()]
        job.input_ids = input_tokens
        job.prompt_tokens = len(input_tokens)
        job.next_step()

    def compute_embed(self, job: Job, logger: Logger, prefill_chunk_size: int):
        if job.compute_step != ComputeStep.EMBED and job.compute_step != ComputeStep.TOKENIZE:
            raise ValueError('Invalid step for embedding')
        if self.input_embedding is None:
            raise RuntimeError("Input Embedding must be loaded before computation")
        
        chunk_start, chunk_end = (0, len(job.input_ids))
        if job.chunking.has_more():
            chunk_start, chunk_end = job.chunking.get_range()
        else:
            chunk_start = chunk_end - 1

        chunk_tokens = job.input_ids[chunk_start:chunk_end]

        comp_state = compute_embedding(
            input_embedder=self.input_embedding, 
            input_ids=tensor([chunk_tokens]), 
            config=self.collector.config, 
            cache=job.cache
        )
        
        job.data = computationStateToJobData(comp_state)
        job.next_step()

    def compute_norm(self, job: Job):
        if job.data is None or job.data.state is None:
            raise RuntimeError("Cannot compute norm without job data")
        norm = self.norm(job.data.state.to(self.device))
        job.set_norm(norm)

    def compute_head(self, job: Job):
        if self.head is None:
            raise RuntimeError("Head must be loaded before computation")
        if job.data is None or job.data.state is None:
            raise RuntimeError("Cannot compute head without job data")
        
        head = compute_head(
            head=self.head, 
            input_ids=job.input_ids, 
            device=self.device, 
            state=job.data.state, 
            top_k=job.top_k,
            top_p=job.top_p,
            min_p=job.min_p,
            temperature=job.temperature,
            presence_penalty=job.presence_penalty
        )

        job.set_output(head, self.collector.config.eos_token_id)
        job.delta = self.tokenizer.decode([job.input_ids[-1]])

    def set_result(self, job: Job):
        res_tokens = job.input_id_tensor()
        if res_tokens is None:
            raise Exception("Cannot decode result tensor: no input ids")
        job.result = self.tokenizer.decode(res_tokens[job.prompt_tokens:])

    def clean_up(self):
        del self.input_embedding
        del self.norm
        del self.head
