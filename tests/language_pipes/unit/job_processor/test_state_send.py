import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from job_processor.util import make_processor, make_job, make_config, FakeModel, FakePipe

class TestSendState(unittest.TestCase):
    """Tests for the _state_send method."""

    def test_transitions_to_done_after_handoff(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_send()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_routes_tokenize_job_to_next_node_when_origin_mismatch(self):
        job = make_job(origin_node_id="node-b")
        job.compute_step = ComputeStep.TOKENIZE
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        pipe = FakePipe(FakeModel("node-c", 0, 0))
        processor = make_processor(
            job=job,
            pipe=pipe,
            config=make_config(node_id="node-a"),
            end_model=None,
        )

        processor.run()

        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-c")
