import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from job_processor.util import make_processor, make_job, FakePipe, FakePipeMulti, FakeModel, TrackingModel, EmptyPipe

class TestProcessLayersState(unittest.TestCase):
    """Tests for the _state_process_layers method."""

    def test_transitions_to_done_when_local_model_missing(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(
            job=job,
            pipe=EmptyPipe(FakeModel("node-a", 0, 0)),
            end_model=None,
        )

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_send_for_remote_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        remote_model = FakeModel("node-b", 1, 1, virtual=True)
        pipe = FakePipeMulti(local_model, remote_model)

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.SEND)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_process_layers_for_local_segment(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        next_model = FakeModel("node-a", 1, 1, virtual=False)
        pipe = FakePipeMulti(local_model, next_model)

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_head_after_all_layers(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = FakePipe(local_model)

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.HEAD)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_embed_for_prefill(self):
        job = make_job()
        job.prompt_tokens = 24
        job.init_chunking(6)
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = FakePipe(local_model)

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_head_after_prefill(self):
        job = make_job()
        job.prompt_tokens = 24
        job.init_chunking(6)
        job.current_chunk = 5
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = FakePipe(local_model)

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)


if __name__ == "__main__":
    unittest.main()