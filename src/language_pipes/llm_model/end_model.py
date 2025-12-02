import os
import torch
from uuid import uuid4
from torch import tensor
from pathlib import Path

from transformers.models.auto.tokenization_auto import AutoTokenizer

from llm_layer_collector import LlmLayerCollector
from llm_layer_collector.auto.auto_rms import AutoRMSNorm
from llm_layer_collector.compute import compute_embedding, compute_head

from language_pipes.util import clone_model
from language_pipes.job_manager.job import ComputeStep, Job
from language_pipes.llm_model.computed import ComputedData
from language_pipes.job_manager.job_data import computationStateToJobData, jobDataToComputationState

class EndModel:
    model_id: str
    process_id: str
    device: str
    input_embedding: torch.nn.Embedding
    norm: AutoRMSNorm
    head: torch.nn.Linear
    collector: LlmLayerCollector

    def __init__(self, app_dir: str, model_id: str, device: str):
        self.model_id = model_id
        self.device = device

        self.process_id = str(uuid4())
        model_dir = str(Path(app_dir) / 'models' / self.model_id)
        if not os.path.exists(model_dir):
            clone_model(model_id, model_dir)
        self.computed = ComputedData(model_dir)
        self.collector = LlmLayerCollector(
            model_dir=os.path.join(model_dir, 'data'),
            cache_file=os.path.join(model_dir, 'cache.json'),
            device=device,
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, 'data'))
    
    def size(self):
        return self.computed.embed_size + self.computed.head_size

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

    def chop_position_embeddings(self, t: torch.Tensor):
        if t is not None:
            return (
                t[0][:, -1:, :],
                t[1][:, -1:, :]
            )

    def compute_embed(self, job: Job):
        if job.current_step != ComputeStep.EMBED:
            self.raise_exception('Invalid step for embedding')
        if self.input_embedding is None:
            self.raise_exception("Input Embedding must be loaded before computation")
        state = None
        if job.data is not None:
            state = jobDataToComputationState(job.data, self.device)
        comp_state = compute_embedding(self.input_embedding, tensor([job.input_ids]).to(self.device), self.collector.config, state)
        if job.current_token > 0:
            comp_state.state = comp_state.state[:, -1:, :]
            if comp_state.causal_mask["full_attention"] is not None:
                comp_state.causal_mask["full_attention"] = comp_state.causal_mask["full_attention"][:, :, -1:, -1:]
            comp_state.position_embeddings = self.chop_position_embeddings(comp_state.position_embeddings)
            comp_state.position_embeddings_local = self.chop_position_embeddings(comp_state.position_embeddings_local)
            comp_state.position_embeddings_global = self.chop_position_embeddings(comp_state.position_embeddings_global)
            
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
        job.set_output(head, self.collector.config.eos_token_id)
        job.delta = self.tokenizer.decode([job.input_ids[-1]])

    def set_result(self, job: Job):
        res_tokens = job.input_id_tensor()
        job.result = self.tokenizer.decode(res_tokens[job.prompt_tokens:])

    def clean_up(self):
        del self.input_embedding
        del self.norm
        del self.head