import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from job_processor.util import make_processor, make_job, FakeEndModel

class TestValidatingState(unittest.TestCase):
    """Tests for the _state_validating method."""

    def test_stops_when_job_missing(self):
        processor = make_processor(job=None, end_model=FakeEndModel())
        next_state = processor._state_validating()
        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_head_when_prefill_done(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 2
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(job=job, end_model=FakeEndModel())
        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.HEAD)

    def test_transitions_to_embed_when_prefill_has_more_chunks(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.current_token = 0
        job.prompt_tokens = 4
        job.init_chunking(chunk_size=2)
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(job=job, end_model=FakeEndModel())
        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.EMBED)

    def test_transitions_to_process_layers_for_local_work(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(job=job, end_model=None)
        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
