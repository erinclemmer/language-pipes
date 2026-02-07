import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep, JobStatus

from util import make_processor, make_job, mock_complete, FakeEndModel, FakeModel, FakeStateNetworkNode, PipeWrapper

class TestFullProcessorRun(unittest.TestCase):
    """End-to-end tests for the full processor run cycle."""

    def test_stops_when_no_pipe_segments(self):
        job = make_job()
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()
        pipe = PipeWrapper("node-1", "model-a", [])
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()
        self.assertEqual(processor.states, [JobState.VALIDATING])
        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(end_model.calls, [])

    def test_stops_when_pipe_incomplete(self):
        job = make_job()
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()
        model = FakeModel("node-1", 0, 0, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [model])
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING])
        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(end_model.calls, [])

    def test_processes_local_layers_and_completes(self):
        job = make_job(complete=mock_complete)
        end_model = FakeEndModel()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-1", "model-a", [model])
        
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.EMBED, JobState.PROCESS_LAYERS, JobState.HEAD])
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertIn("compute_head", end_model.calls)

    def test_sends_job_to_virtual_segment(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        virtual_model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [virtual_model, local_model])
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.SEND])
        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(pipe.calls, ["send job"])

    def test_missing_model_stops_processing(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-1", "model-a", [model])
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING])
        self.assertEqual(processor.state, JobState.DONE)

if __name__ == "__main__":
    unittest.main()