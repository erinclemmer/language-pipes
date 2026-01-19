import cProfile
import pstats
import io
import torch
import sys
from typing import Optional
from dataclasses import dataclass
from logging import Logger

# Add src to path so we can import language_pipes
sys.path.append('src')

from language_pipes.config import LpConfig
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobContext, JobProcessor, JobState
from language_pipes.util.enums import ComputeStep, JobStatus

# --- Mocks (copied/adapted from tests/unit/job_processor/util.py) ---

class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def error(self, message):
        self.messages.append(("error", message))

class FakeEndModel:
    def __init__(self):
        self.calls = []

    def tokenize(self, job):
        self.calls.append("tokenize")
        job.input_ids = list(range(24))
        job.prompt_tokens = len(job.input_ids)
        job.next_step()

    def compute_embed(self, job, logger=None, chunk_size=None, chunk_start=0, chunk_end=-1):
        self.calls.append("compute_embed")
        if job.data is None:
            job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.next_step()

    def compute_norm(self, job):
        self.calls.append("compute_norm")
        if job.data:
            job.set_norm(job.data.state)

    def compute_head(self, job):
        self.calls.append("compute_head")
        # Emit token 1, eos is 0. So it continues.
        # But we need it to eventually stop.
        # Job stops if token == eos_token OR max_completion_tokens reached.
        # Job handles max_completion_tokens in next_step() but set_output checks eos.
        job.set_output(token=1, eos_token=0)

    def set_result(self, job):
        self.calls.append("set_result")
        job.result = "done"

class FakeModel:
    def __init__(self, node_id, start_layer, end_layer, virtual=False, num_hidden_layers=1):
        self.node_id = node_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.virtual = virtual
        self.loaded = True
        self.num_hidden_layers = num_hidden_layers

    def process_job(self, job):
        if job.data is None:
            job.data = JobData()
        # Simulate some computation
        torch.matmul(torch.randn(500, 500), torch.randn(500, 500))
        job.set_layer(torch.zeros((1, 1)), self.end_layer + 1, self.num_hidden_layers)

class FakePipe:
    def __init__(self, model):
        self._model = model
        self.sent_jobs = []

    def is_complete(self):
        return True

    def get_layer(self, layer, need_physical=False):
        return self._model

    def send_job(self, job, node_id):
        self.sent_jobs.append((job, node_id))

def make_config(node_id="node-a", prefill_chunk_size=2, print_times=False, print_job_data=False):
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        model_dir="./models",
        oai_port=None,
        node_id=node_id,
        hosted_models=[],
        max_pipes=1,
        model_validation=False,
        print_times=print_times,
        print_job_data=print_job_data,
        prefill_chunk_size=prefill_chunk_size,
    )

def make_job(**kwargs):
    defaults = {
        "origin_node_id": "node-a",
        "messages": [],
        "pipe_id": "pipe-1",
        "model_id": "model-1",
        "max_completion_tokens": 5, # Generate a few tokens
    }
    defaults.update(kwargs)
    job = Job(**defaults)
    # Mock update/complete callbacks
    job.update = lambda j: True
    job.complete = lambda: None
    return job

# --- Profiling Scenario ---

def run_scenario():
    # Setup
    config = make_config()
    logger = FakeLogger()
    # A model that handles layers 0 to 5 locally
    model = FakeModel(node_id="node-a", start_layer=0, end_layer=5, num_hidden_layers=6)
    pipe = FakePipe(model)
    end_model = FakeEndModel()
    
    # Create a job that starts at HEAD
    job = make_job()
    job.compute_step = ComputeStep.HEAD
    job.current_layer = 0
    job.input_ids = [1, 2, 3] # Some initial tokens
    job.prompt_tokens = 3
    job.data = JobData()
    job.data.state = torch.zeros((1, 1))
    
    # Initialize chunking so HEAD state doesn't fail on "not done chunking" or similar checks
    # If compute_step is HEAD, it usually means we are done with prefill or starting decode?
    # Looking at _state_head:
    # if job.current_token == 0:
    #     if job.chunking.has_more(): ...
    
    # Let's say we are starting generation.
    job.init_chunking(chunk_size=100) # Large chunk size so no chunking needed
    
    ctx = JobContext(
        job=job,
        pipe=pipe,
        logger=logger,
        config=config,
        end_model=end_model
    )
    
    processor = JobProcessor(ctx)
    
    # Run the processor until it hits DONE
    # Note: JobProcessor.run() loops until DONE.
    # However, our FakePipe/Model setup might transition to SEND or similar which leads to DONE.
    # In a real loop, the job would come back. 
    # Here JobProcessor handles one "pass" (e.g. HEAD -> EMBED -> LAYERS -> SEND/DONE).
    # To really stress it, we might want to run it multiple times or have a job that stays local.
    
    # Let's make the model local for all layers so it loops a bit.
    # JobProcessor state machine:
    # HEAD -> EMBED
    # EMBED -> PROCESS_LAYERS (if local)
    # PROCESS_LAYERS -> PROCESS_LAYERS (if next layer local)
    # ...
    # eventually SEND or DONE.
    
    processor.run()

def profile_it():
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the scenario multiple times to get better stats
    for _ in range(100):
        run_scenario()
        
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50) # Print top 50
    print(s.getvalue())

if __name__ == "__main__":
    profile_it()
