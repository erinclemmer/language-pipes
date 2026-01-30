import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep, JobStatus

from job_processor.util import make_processor, make_job, mock_complete, FakeEndModel, FakeModel, FakePipe, IncompletePipe, EmptyPipe

class TestFullProcessorRun(unittest.TestCase):
    """End-to-end tests for the full processor run cycle."""

    def test_stops_when_pipe_incomplete(self):
        job = make_job()
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()

        pipe = IncompletePipe(FakeModel("node-a", 0, 0))
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(end_model.calls, [])

    def test_processes_local_layers_and_completes(self):
        job = make_job(complete=mock_complete)
        end_model = FakeEndModel()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = FakePipe(model)

        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertIn("compute_head", end_model.calls)

    def test_sends_job_to_virtual_segment(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=1)
        pipe = FakePipe(model)
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        processor.run()

        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_missing_model_stops_processing(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(
            job=job,
            pipe=EmptyPipe(FakeModel("node-a", 0, 0)),
            end_model=None,
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)

if __name__ == "__main__":
    unittest.main()