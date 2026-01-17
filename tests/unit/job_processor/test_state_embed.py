import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from language_pipes.config import LpConfig
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobContext, JobProcessor, JobState
from language_pipes.util.enums import ComputeStep, JobStatus

from tests.unit.job_processor.util import make_processor, make_job, make_config, FakeEndModel, FakeModel, FakePipe, EmptyPipe

class TestEmbedState(unittest.TestCase):
    """Tests for the _state_embed method."""

    def test_transitions_to_done_when_update_fails(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update)
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 2
        job.current_token = 0
        job.init_chunking(chunk_size=1)

        processor = make_processor(
            job=job,
            config=make_config(prefill_chunk_size=1),
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(updates), 1)

    def test_transitions_to_done_when_model_missing(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        processor = make_processor(
            job=job,
            pipe=EmptyPipe(FakeModel("node-a", 0, 0)),
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_send_for_remote_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = make_processor(job=job, pipe=pipe, end_model=FakeEndModel())

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.SEND)

    def test_transitions_to_process_layers_for_local_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        processor = make_processor(
            job=job,
            pipe=FakePipe(FakeModel("node-a", 0, 0, virtual=False)),
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_process_layers_for_prefill(self):
        job = make_job()
        job._test_input_ids = 24
        job.compute_step = ComputeStep.EMBED

        processor = make_processor(
            job=job,
            pipe=FakePipe(FakeModel("node-a", 0, 0, virtual=False)),
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
        chunk_start, chunk_end = job.chunking.get_range()
        self.assertEqual(chunk_start, 0)
        self.assertEqual(chunk_end, 2)

        job.compute_step = ComputeStep.EMBED
        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
        chunk_start, chunk_end = job.chunking.get_range()
        self.assertEqual(chunk_start, 2)
        self.assertEqual(chunk_end, 4)

class TestEmbedPrefillIntegration(unittest.TestCase):
    """Integration tests for embed state during prefill operations."""

    def test_update_failure_stops(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update)
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()

        processor = make_processor(
            job=job,
            config=make_config(prefill_chunk_size=1),
            end_model=end_model,
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)
        self.assertIn("tokenize", end_model.calls)
        self.assertIn("compute_embed", end_model.calls)
        self.assertEqual(len(updates), 1)