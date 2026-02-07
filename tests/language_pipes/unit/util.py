import torch
from typing import Optional

from language_pipes.pipes.pipe import Pipe
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
    def __init__(self, num_local_layers: int = 0):
        self.calls = []
        self.layers = list(range(num_local_layers))

    def tokenize(self, job):
        self.calls.append("tokenize")
        job.input_ids = list(range(2))
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

def make_config(node_id="node-a", prefill_chunk_size=2):
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        model_dir="./models",
        oai_port=None,
        node_id=node_id,
        layer_models=[],
        end_models=None,
        max_pipes=1,
        model_validation=False,
        prefill_chunk_size=prefill_chunk_size,
        num_local_layers=0,
        huggingface_token=None
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


class ProcessorWrapper(JobProcessor):
    def __init__(self, ctx: JobContext):
        super().__init__(ctx)
        self.states = []
    
    def _transition(self):
        self.states.append(self.state)
        return super()._transition()

def make_processor(job=None, pipe: Optional[Pipe]=None, end_model=None, config=None, logger=None):
    """Helper to create a JobProcessor with sensible defaults."""
    return ProcessorWrapper(
        JobContext(
            config=config or make_config(),
            logger=logger or FakeLogger(),
            job=job,
            pipe=pipe,
            end_model=end_model,
        )
    )


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeConnection:
    def __init__(self, address: str):
        self.address = address


class FakeStateNetworkNode:
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._data = {}
        self._peers = []
        self.logger = None

    def node_id(self):
        return self._node_id

    def read_data(self, node_id: str, key: str):
        return self._data.get((node_id, key))

    def update_data(self, key: str, value: str):
        self._data[(self._node_id, key)] = value

    def peers(self):
        return self._peers

    def connection_from_node(self, node_id: str):
        return FakeConnection("127.0.0.1")

    def add_peer(self, peer_id: str, models=None):
        self._peers.append(peer_id)
        if models is None:
            models = []
        import json
        self._data[(peer_id, "models")] = json.dumps([m.to_json() for m in models])

class PipeWrapper(Pipe):
    def __init__(self, node_id: str, model_id: str, segments: List[FakeModel]):
        router = FakeStateNetworkNode(node_id)
        super().__init__(router, None, model_id, "")
        self.calls = []
        self.segments = segments
    
    def tokenizer():
        self.calls.append("tokenizer")
        def t(x):
            return "res"
        return t
    
    def send_job(self, job, node_id):
        self.calls.append("send job")

    def empty():
        pass