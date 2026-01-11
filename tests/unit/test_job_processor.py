import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.config import LpConfig
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobContext, JobProcessor, JobState
from language_pipes.network.config import NetworkConfig
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
        job.input_ids = [1, 2]
        job.prompt_tokens = len(job.input_ids)
        job.next_step()

    def compute_embed(self, job, chunk_start=0, chunk_end=-1):
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
        job.set_layer(torch.zeros((1, 1)), self.end_layer + 1)
        if job.current_layer == self.num_hidden_layers:
            job.compute_step = ComputeStep.HEAD


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


def make_config(node_id="node-a"):
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        oai_port=None,
        router=NetworkConfig(provider="dsn", settings={"node_id": node_id}),
        node_id=node_id,
        hosted_models=[],
        job_port=0,
        max_pipes=1,
        model_validation=False,
        ecdsa_verification=False,
        print_times=False,
        print_job_data=False,
        prefill_chunk_size=2,
    )


class JobProcessorTests(unittest.TestCase):
    def test_stops_when_pipe_incomplete(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.TOKENIZE
        logger = FakeLogger()
        end_model = FakeEndModel()

        class IncompletePipe(FakePipe):
            def is_complete(self):
                return False

        pipe = IncompletePipe(FakeModel("node-a", 0, 0))
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=logger,
                job=job,
                pipe=pipe,
                end_model=end_model,
            )
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(end_model.calls, [])

    def test_processes_local_layers_and_completes(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        logger = FakeLogger()
        end_model = FakeEndModel()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = FakePipe(model)

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=logger,
                job=job,
                pipe=pipe,
                end_model=end_model,
            )
        )

        processor.run()

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertIn("compute_head", end_model.calls)

    def test_sends_job_to_virtual_segment(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=1)
        pipe = FakePipe(model)
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=None,
            )
        )

        processor.run()

        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_head_origin_mismatch_stops(self):
        job = Job(origin_node_id="node-b", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = JobProcessor(
            JobContext(
                config=make_config(node_id="node-a"),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)

    def test_missing_model_stops_processing(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        class EmptyPipe(FakePipe):
            def get_layer(self, layer, need_physical=False):
                return None

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=EmptyPipe(FakeModel("node-a", 0, 0)),
                end_model=None,
            )
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)


if __name__ == "__main__":
    unittest.main()
