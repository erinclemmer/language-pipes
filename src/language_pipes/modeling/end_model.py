import ctypes
import gc
import os
import torch
from uuid import uuid4
from torch import tensor
from pathlib import Path

from transformers.cache_utils import DynamicCache
from transformers.models.auto.tokenization_auto import AutoTokenizer

from llm_layer_collector import LlmLayerCollector
from llm_layer_collector.auto.auto_rms import AutoRMSNorm
from llm_layer_collector.compute import compute_embedding

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

    def compute_embed(self, job: Job, cache: DynamicCache, chunk_start: int = 0, chunk_end: int = -1):
        """
        Compute embeddings for a job, optionally for a specific chunk.
        
        Args:
            job: The job to process
            cache: The KV cache (DynamicCache)
            chunk_start: Start index in input_ids for this chunk (0 for no chunking)
            chunk_end: End index (exclusive) in input_ids (-1 means use full sequence)
        """
        if job.current_step != ComputeStep.EMBED:
            raise ValueError('Invalid step for embedding')
        if self.input_embedding is None:
            raise RuntimeError("Input Embedding must be loaded before computation")
        
        # Determine which tokens to embed and whether this is chunked prefill
        is_chunked_prefill = False
        if job.current_token == 0:
            # Prefill phase - may be chunked
            if chunk_end == -1:
                chunk_end = len(job.input_ids)
            chunk_tokens = job.input_ids[chunk_start:chunk_end]
            # Chunked prefill: processing a non-first chunk during prefill
            is_chunked_prefill = chunk_start > 0
        else:
            # Decode phase - always single token, no chunking
            chunk_tokens = [job.input_ids[-1]]
        
        # Compute embeddings for the chunk
        # chunked_prefill=True tells compute_embedding to process all tokens
        # even when cache is non-empty (for subsequent prefill chunks)
        comp_state = compute_embedding(
            self.input_embedding, 
            tensor([chunk_tokens]).to(self.device), 
            self.collector.config, 
            cache,
            chunked_prefill=is_chunked_prefill
        )
        
        job.data = computationStateToJobData(comp_state)
        job.data_hash = job.data.hash_state()
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
        with torch.inference_mode():
            state_on_device = job.data.state.detach().clone()[:, -1, :].to(self.device)
            logits = torch.nn.functional.linear(
                state_on_device, 
                self.head.weight, 
                self.head.bias
            ).flatten()
            del state_on_device
            
            # Apply presence penalty to discourage token repetition
            # Subtracts penalty from logits of tokens that have already appeared
            if job.presence_penalty != 0 and len(job.input_ids) > 0:
                unique_tokens = set(job.input_ids)
                for token_id in unique_tokens:
                    logits[token_id] -= job.presence_penalty

            if job.temperature == 0:
                # Greedy decoding - just pick the top token
                head = int(logits.argmax().item())
            else:
                # Apply temperature scaling to logits before softmax
                # Lower temperature = sharper distribution (more deterministic)
                # Higher temperature = flatter distribution (more random)
                scaled_logits = logits / job.temperature

                # Apply min_p filtering if specified (min_p > 0)
                # Remove tokens with probability < min_p * max_probability
                if job.min_p > 0:
                    probs = torch.nn.functional.softmax(scaled_logits, dim=0)
                    max_prob = probs.max()
                    min_prob_threshold = job.min_p * max_prob
                    indices_to_remove = probs < min_prob_threshold
                    scaled_logits[indices_to_remove] = float('-inf')

                # Apply top_p (nucleus) filtering if specified (top_p < 1.0)
                # Mask out tokens outside the top-p cumulative probability mass
                if job.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=0)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    # Find indices to remove (cumulative prob exceeds top_p)
                    sorted_indices_to_remove = cumulative_probs > job.top_p
                    # Shift to keep at least one token
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    # Scatter -inf back to original positions
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = float('-inf')

                # Apply top_k filtering if specified (top_k > 0)
                # Mask out non-top-k tokens by setting them to -inf
                if job.top_k > 0:
                    k = min(job.top_k, scaled_logits.size(0))
                    top_k_values, _ = torch.topk(scaled_logits, k)
                    threshold = top_k_values[-1]
                    scaled_logits = torch.where(scaled_logits < threshold, torch.tensor(float('-inf'), device=scaled_logits.device), scaled_logits)

                probabilities = torch.nn.functional.softmax(scaled_logits, dim=0)
                head = int(torch.multinomial(probabilities, num_samples=1).item())
            
            del logits
        
        job.set_output(head, self.collector.config.eos_token_id)
        job.delta = self.tokenizer.decode([job.input_ids[-1]])

    def set_result(self, job: Job):
        res_tokens = job.input_id_tensor()
        job.result = self.tokenizer.decode(res_tokens[job.prompt_tokens:])

    def clean_up(self):
        del self.input_embedding
        del self.norm
        del self.head
