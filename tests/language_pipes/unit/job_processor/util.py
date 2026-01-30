import torch
from typing import Optional

from language_pipes.config import LpConfig
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobContext, JobProcessor, JobState
from language_pipes.util.enums import ComputeStep, JobStatus

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

    def compute_embed(self, job, chunk_start=0, chunk_end=-1):
        if job.compute_step == ComputeStep.TOKENIZE:
            self.calls.append("tokenize")
        self.calls.append("compute_embed")
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.next_step()

    def compute_norm(self, job):
        self.calls.append("compute_norm")
        job.set_norm(job.data.state)

    def compute_head(self, job):
        self.calls.append("compute_head")
        job.set_output(token=0, eos_token=0)

    def set_result(self, job):
        self.calls.append("set_result")
        job.result = "done"


class FakeEndModelContinue(FakeEndModel):
    def compute_head(self, job):
        self.calls.append("compute_head")
        job.set_output(token=1, eos_token=0)


class FakeModel:
    def __init__(self, node_id, start_layer, end_layer, virtual=False, num_hidden_layers=1):
        self.node_id = node_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.virtual = virtual
        self.loaded = True
        self.num_hidden_layers = num_hidden_layers

    def process_job(self, job, logger):
        if job.data is None:
            job.data = JobData()
        job.set_layer(torch.zeros((1, 1)), self.end_layer + 1, self.num_hidden_layers)

class TrackingModel(FakeModel):
    def __init__(self, node_id, start_layer, end_layer, virtual=False, num_hidden_layers=1):
        super().__init__(node_id, start_layer, end_layer, virtual=virtual, num_hidden_layers=num_hidden_layers)
        self.processed = False

    def process_job(self, job, logger):
        self.processed = True
        super().process_job(job, logger)

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


class FakePipeMulti(FakePipe):
    def __init__(self, local_model, next_model):
        super().__init__(local_model)
        self._local_model = local_model
        self._next_model = next_model

    def get_layer(self, layer, need_physical=False):
        if need_physical:
            return self._local_model
        return self._next_model


class EmptyPipe(FakePipe):
    def get_layer(self, layer, need_physical=False):
        return None


class IncompletePipe(FakePipe):
    def is_complete(self):
        return False


def make_config(node_id="node-a", prefill_chunk_size=2):
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        model_dir="./models",
        oai_port=None,
        node_id=node_id,
        hosted_models=[],
        max_pipes=1,
        model_validation=False,
        prefill_chunk_size=prefill_chunk_size,
    )

def mock_complete(a):
    pass


def make_job(**kwargs):
    """Helper to create a Job with sensible defaults."""
    defaults = {
        "origin_node_id": "node-a",
        "messages": [],
        "pipe_id": "pipe-1",
        "model_id": "model-1",
        "prefill_chunk_size": 6
    }
    defaults.update(kwargs)
    return Job(**defaults)


def make_processor(job=None, pipe: Optional[FakePipe]=None, end_model=None, config=None, logger=None):
    """Helper to create a JobProcessor with sensible defaults."""
    return JobProcessor(
        JobContext(
            config=config or make_config(),
            logger=logger or FakeLogger(),
            job=job,
            pipe=pipe or FakePipe(FakeModel("node-a", 0, 0)),
            end_model=end_model,
        )
    )
