import os
import sys
import torch
import unittest
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep, JobStatus

from job_processor.util import mock_complete, make_processor, make_job, make_config, FakeEndModel, FakeModel, FakePipe, FakeEndModelContinue, FakeLogger

class TestHeadState(unittest.TestCase):
    """Tests for the _state_head method."""

    def test_transitions_to_done_on_completion(self):
        completed = []

        def mark_complete(job):
            completed.append(job.job_id)

        job = make_job(complete=mark_complete)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModel()
        processor = make_processor(job=job, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIn("set_result", end_model.calls)
        self.assertEqual(completed, [job.job_id])

    def test_transitions_to_done_on_update_failure(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        processor = make_processor(job=job, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(job.status, JobStatus.IN_PROGRESS)
        self.assertEqual(len(updates), 1)
        self.assertNotIn("set_result", end_model.calls)

    def test_transitions_to_embed_on_successful_update(self):
        updates = []

        def record_update(job):
            updates.append(job.compute_step)
            return True

        job = make_job(update=record_update)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 0
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        processor = make_processor(job=job, end_model=end_model)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertEqual(len(updates), 1)
        self.assertIn("compute_norm", end_model.calls)
        self.assertIn("compute_head", end_model.calls)

    def test_origin_mismatch_stops(self):
        job = make_job(origin_node_id="node-b")
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        processor = make_processor(
            job=job,
            config=make_config(node_id="node-a"),
            end_model=FakeEndModel(),
        )

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)

    def test_logs_job_data_and_timing_when_enabled(self):
        job = make_job(complete=mock_complete)
        job.compute_step = ComputeStep.HEAD
        job.current_layer = 1
        job.prompt_tokens = 2
        job.prefill_start_time = time()
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))
        logger = FakeLogger()

        processor = make_processor(
            job=job,
            config=make_config(),
            logger=logger,
            end_model=FakeEndModel(),
        )

        processor.run()

        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertTrue(any("Timing" in message for _, message in logger.messages))
        self.assertTrue(any("Job ID" in message for _, message in logger.messages))

class TestHeadFlowIntegration(unittest.TestCase):
    """Integration tests for head state transitions through subsequent states."""

    def test_sends_to_remote_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        pipe = FakePipe(FakeModel("node-b", 0, 0, virtual=True))
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.SEND)

        final_state = processor._state_send()
        self.assertEqual(final_state, JobState.DONE)
        self.assertEqual(len(pipe.sent_jobs), 1)
        _, node_id = pipe.sent_jobs[0]
        self.assertEqual(node_id, "node-b")

    def test_processes_local_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.prompt_tokens = 1
        job.data = JobData()
        job.data.state = torch.zeros((1, 1))

        end_model = FakeEndModelContinue()
        pipe = FakePipe(FakeModel("node-a", 0, 0, virtual=False))
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        head_state = processor._state_head()
        self.assertEqual(head_state, JobState.EMBED)

        next_state = processor._state_embed()
        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
