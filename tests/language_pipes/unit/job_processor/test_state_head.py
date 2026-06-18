import os
import sys
import unittest
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep, JobStatus

from util import mock_complete, make_processor, make_job, make_job_data, FakeEndModel, FakeModel, FakeEndModelContinue, PipeWrapper

class TestHeadState(unittest.TestCase):
    """Tests for the _state_head method."""

    def test_transitions_to_done_on_completion(self):
        completed = []

        def mark_complete(job):
            completed.append(job.job_id)

        job = make_job(complete=mark_complete)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = make_job_data()

        end_model = FakeEndModel()
        processor = make_processor(job=job, pipe=None, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIn("set_result", end_model.calls)
        self.assertEqual(completed, [job.job_id])

    def test_transitions_to_done_on_update_failure(self):
        updates = []
        completed = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        def mark_complete(job):
            completed.append(job.job_id)

        job = make_job(update=fail_update, complete=mark_complete)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = make_job_data()

        end_model = FakeEndModelContinue()
        processor = make_processor(job=job, pipe=None, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(len(updates), 1)
        self.assertIn("set_result", end_model.calls)
        self.assertEqual(completed, [job.job_id])

    def test_transitions_to_embed_on_successful_update(self):
        updates = []

        def record_update(job):
            updates.append(job.compute_step)
            return True

        job = make_job(update=record_update)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = make_job_data()

        end_model = FakeEndModelContinue()
        processor = make_processor(job=job, pipe=None, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertEqual(len(updates), 1)
        self.assertIn("compute_norm", end_model.calls)
        self.assertIn("compute_head", end_model.calls)

    def test_origin_mismatch_stops(self):
        job = make_job(origin_node_id="node-b")
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.data = make_job_data()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])

        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING])
        self.assertEqual(processor.state, JobState.DONE)

    def test_completes_without_logging_dependencies(self):
        job = make_job(complete=mock_complete)
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.prompt_tokens = 2
        job.data = make_job_data()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])

        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.HEAD])
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")

class TestHeadFlowIntegration(unittest.TestCase):
    """Integration tests for head state transitions through subsequent states."""

    def test_sends_to_remote_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = make_job_data()

        end_model = FakeEndModelContinue()
        virtual_model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.SEND)

        final_state = processor._state_send()
        self.assertEqual(final_state, JobState.DONE)
        self.assertEqual(pipe.calls, ["send job"])

    def test_processes_local_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = make_job_data()

        end_model = FakeEndModelContinue()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
