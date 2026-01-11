import os
import sys
import unittest
from time import time

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

    def process_job(self, job):
        if job.data is None:
            job.data = JobData()
        job.set_layer(torch.zeros((1, 1)), self.end_layer + 1)
        if job.current_layer == self.num_hidden_layers:
            job.compute_step = ComputeStep.HEAD


class TrackingModel(FakeModel):
    def __init__(self, node_id, start_layer, end_layer, virtual=False, num_hidden_layers=1):
        super().__init__(node_id, start_layer, end_layer, virtual=virtual, num_hidden_layers=num_hidden_layers)
        self.processed = False

    def process_job(self, job):
        self.processed = True
        super().process_job(job)


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


def make_config(node_id="node-a", prefill_chunk_size=2, print_times=False, print_job_data=False):
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        model_dir="./models",
        oai_port=None,
        router=NetworkConfig(provider="dsn", settings={"node_id": node_id}),
        node_id=node_id,
        hosted_models=[],
        job_port=0,
        max_pipes=1,
        model_validation=False,
        ecdsa_verification=False,
        print_times=print_times,
        print_job_data=print_job_data,
        prefill_chunk_size=prefill_chunk_size,
    )


def mock_complete(a):
    pass

class JobProcessorTests(unittest.TestCase):
    def test_validating_stops_when_job_missing(self):
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=None,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.DONE)

    def test_validating_transitions_to_head_when_prefill_done(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 2
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.HEAD)

    def test_validating_transitions_to_embed_when_prefill_has_more_chunks(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.HEAD
        job.current_token = 0
        job.prompt_tokens = 4
        job.init_chunking(chunk_size=2)
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.EMBED)

    def test_validating_transitions_to_process_layers_for_local_work(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=None,
            )
        )

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

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
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1", complete=mock_complete)
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

    def test_head_transitions_to_done_on_completion(self):
        completed = []

        def mark_complete(job):
            completed.append(job.job_id)

        job = Job(
            origin_node_id="node-a",
            messages=[],
            pipe_id="pipe-1",
            model_id="model-1",
            complete=mark_complete,
        )
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModel()
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=end_model,
            )
        )

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIn("set_result", end_model.calls)
        self.assertEqual(completed, [job.job_id])

    def test_head_transitions_to_done_on_update_failure(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = Job(
            origin_node_id="node-a",
            messages=[],
            pipe_id="pipe-1",
            model_id="model-1",
            update=fail_update,
        )
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=end_model,
            )
        )

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.IN_PROGRESS)
        self.assertEqual(len(updates), 1)
        self.assertNotIn("set_result", end_model.calls)

    def test_head_transitions_to_embed_on_successful_update(self):
        updates = []

        def record_update(job):
            updates.append(job.compute_step)
            return True

        job = Job(
            origin_node_id="node-a",
            messages=[],
            pipe_id="pipe-1",
            model_id="model-1",
            update=record_update,
        )
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=end_model,
            )
        )

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertEqual(len(updates), 1)
        self.assertIn("compute_norm", end_model.calls)
        self.assertIn("compute_head", end_model.calls)

    def test_head_flow_sends_to_remote_layer(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=end_model,
            )
        )

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.SEND)

        final_state = processor._state_send()
        self.assertEqual(final_state, JobState.DONE)
        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_head_flow_processes_local_layer(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        pipe = FakePipe(FakeModel("node-a", 0, 0, virtual=False))
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=end_model,
            )
        )

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_embed_prefill_update_failure_stops(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1", update=fail_update)
        job.compute_step = ComputeStep.TOKENIZE
        logger = FakeLogger()
        end_model = FakeEndModel()

        processor = JobProcessor(
            JobContext(
                config=make_config(prefill_chunk_size=1),
                logger=logger,
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=end_model,
            )
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)
        self.assertIn("tokenize", end_model.calls)
        self.assertIn("compute_embed", end_model.calls)
        self.assertEqual(len(updates), 1)

    def test_embed_transitions_to_done_when_update_fails(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = Job(
            origin_node_id="node-a",
            messages=[],
            pipe_id="pipe-1",
            model_id="model-1",
            update=fail_update,
        )
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 2
        job.current_token = 0
        job.init_chunking(chunk_size=1)

        processor = JobProcessor(
            JobContext(
                config=make_config(prefill_chunk_size=1),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(updates), 1)

    def test_embed_transitions_to_done_when_model_missing(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        class EmptyPipe(FakePipe):
            def get_layer(self, layer, need_physical=False):
                return None

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=EmptyPipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)

    def test_embed_transitions_to_send_for_remote_layer(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.SEND)

    def test_embed_transitions_to_process_layers_for_local_layer(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0, virtual=False)),
                end_model=FakeEndModel(),
            )
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_process_layers_transitions_to_done_when_local_model_missing(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
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

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.DONE)

    def test_process_layers_transitions_to_send_for_remote_layer(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        remote_model = FakeModel("node-b", 1, 1, virtual=True)
        pipe = FakePipeMulti(local_model, remote_model)

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=None,
            )
        )

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.SEND)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_process_layers_transitions_to_process_layers_for_local_segment(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        next_model = FakeModel("node-a", 1, 1, virtual=False)
        pipe = FakePipeMulti(local_model, next_model)

        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=None,
            )
        )

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_send_transitions_to_done_after_handoff(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = JobProcessor(
            JobContext(
                config=make_config(),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=None,
            )
        )

        next_state = processor._state_send()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_head_logs_job_data_and_timing_when_enabled(self):
        job = Job(origin_node_id="node-a", messages=[], pipe_id="pipe-1", model_id="model-1", complete=mock_complete)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.prompt_tokens = 2
        job.prefill_start_time = time()
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        logger = FakeLogger()

        processor = JobProcessor(
            JobContext(
                config=make_config(print_times=True, print_job_data=True),
                logger=logger,
                job=job,
                pipe=FakePipe(FakeModel("node-a", 0, 0)),
                end_model=FakeEndModel(),
            )
        )

        processor.run()

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertTrue(any("Timing" in message for _, message in logger.messages))
        self.assertTrue(any("Job ID" in message for _, message in logger.messages))

    def test_send_routes_tokenize_job_to_next_node_when_origin_mismatch(self):
        job = Job(origin_node_id="node-b", messages=[], pipe_id="pipe-1", model_id="model-1")
        job.compute_step = ComputeStep.TOKENIZE
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        pipe = FakePipe(FakeModel("node-c", 0, 0))
        processor = JobProcessor(
            JobContext(
                config=make_config(node_id="node-a"),
                logger=FakeLogger(),
                job=job,
                pipe=pipe,
                end_model=None,
            )
        )

        processor.run()

        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-c")

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
