import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_processor import JobState
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.enums import ComputeStep

from util import (
    make_processor,
    make_job,
    mock_complete,
    FakeEndModel,
    FakeEndModelContinue,
    FakeModel,
    PipeWrapper,
)

class TestEmbedState(unittest.TestCase):
    """Tests for the _state_embed method."""

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_transitions_to_done_when_update_fails(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update)
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 2
        job.current_token = 0
        job.init_chunking()

        processor = make_processor(
            job=job,
            pipe=None,
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(updates), 1)

    def test_transitions_to_done_when_model_missing(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_send_for_remote_layer(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        virtual_model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(job=job, pipe=pipe, end_model=FakeEndModel())

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.SEND)

    def test_transitions_to_process_layers_for_local_layer(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_process_layers_for_prefill(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=0),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

        job.compute_step = ComputeStep.EMBED
        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_process_layers_for_num_local_layers(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED

        virtual_model = FakeModel("node-a", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(
            job=job,
            # Simulate node loading the same layer as the EndModel
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=1)
        )

        next_state = processor._state_embed()

        # The job should not be sent to the virtual node
        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_send_without_local_layers(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        virtual_model = FakeModel("node-a", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(
            job=job,
            # Simulate node loading the same layer as the EndModel
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=0)
        )

        next_state = processor._state_embed()

        # The job should be sent to the virtual node if no local layers are specified
        self.assertEqual(next_state, JobState.SEND)

    def test_transitions_to_send_for_misaligned_layers(self):
        """Allow starting computation for a layer not at the start layer of a model"""
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.EMBED
        virtual_model = FakeModel("node-a", 0, 1, True, 2)
        end_model = FakeEndModel(num_local_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model])

        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=end_model
        )

        processor.run()

        self.assertEqual(processor.states, 
            [JobState.VALIDATING, JobState.EMBED, JobState.PROCESS_LAYERS, JobState.SEND]
        )
        self.assertEqual(job.current_layer, 1)

class TestEmbedPrefillIntegration(unittest.TestCase):
    """Integration tests for embed state during prefill operations."""

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_update_failure_stops(self):
        updates = []

        def complete(_):
            pass

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update, complete=complete)
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])

        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=end_model,
        )

        processor.run()

        self.assertEqual(
            processor.states,
            [JobState.VALIDATING, JobState.EMBED, JobState.PROCESS_LAYERS, JobState.EMBED],
        )
        self.assertEqual(processor.state, JobState.DONE)
        self.assertIn("tokenize", end_model.calls)
        self.assertIn("compute_embed", end_model.calls)

class TestRestartAtTheOrigin(unittest.TestCase):
    """A node that cannot validate a packet strips it and bounces it back. The
    origin must resend the pass, not embed again: `_state_embed` advances the
    prefill chunk, which would skip it, and embedding a decode token again
    would put it into the caches of every node before the corruption point a
    second time."""

    def make_origin(self, end_model):
        job = make_job(complete=mock_complete)
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.TOKENIZE

        # The layers live on another node, so the origin has to send the job out
        remote = FakeModel("node-b", 0, 1, virtual=True, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [remote])
        return job, pipe

    def dispatch(self, job, pipe, end_model):
        make_processor(job=job, pipe=pipe, end_model=end_model).run()

    def return_from_pipe(self, job, pipe):
        """Hand the pass back the way the last node on the pipe would."""
        sent = pipe.sent_jobs[-1]
        job.receive_network_job(NetworkJob(
            job_id=sent.job_id,
            pipe_id=sent.pipe_id,
            origin_node_id=sent.origin_node_id,
            current_layer=0,
            data=sent.data,
            data_hash=b"",
            compute_step=ComputeStep.HEAD,
            times=[],
            pass_idx=sent.pass_idx
        ), job.origin_node_id)

    def bounce(self, job, pipe, pass_idx=None):
        """The packet `JobReceiver.restart_token` sends back to the origin."""
        sent = pipe.sent_jobs[-1]
        return job.receive_network_job(NetworkJob(
            job_id=sent.job_id,
            pipe_id=sent.pipe_id,
            origin_node_id=sent.origin_node_id,
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.EMBED,
            times=[],
            pass_idx=sent.pass_idx if pass_idx is None else pass_idx
        ), job.origin_node_id)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_a_bounced_prefill_chunk_is_resent_not_skipped(self):
        # 2 prompt tokens at a chunk size of 1 => two prefill chunks
        job, pipe = self.make_origin(FakeEndModel())
        self.dispatch(job, pipe, FakeEndModel())
        sent = pipe.sent_jobs[0]
        chunk = job.chunking.current_chunk

        self.assertTrue(self.bounce(job, pipe))

        end_model = FakeEndModel()
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)
        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.SEND])
        self.assertEqual(end_model.calls, [])
        self.assertEqual(job.chunking.current_chunk, chunk)
        self.assertEqual(len(pipe.sent_jobs), 2)
        self.assertIs(pipe.sent_jobs[1].data, sent.data)
        self.assertEqual(pipe.sent_jobs[1].pass_idx, sent.pass_idx)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_the_next_chunk_still_advances_after_a_restart(self):
        job, pipe = self.make_origin(FakeEndModel())
        self.dispatch(job, pipe, FakeEndModel())
        self.bounce(job, pipe)
        self.dispatch(job, pipe, FakeEndModel())

        self.return_from_pipe(job, pipe)
        end_model = FakeEndModel()
        self.dispatch(job, pipe, end_model)

        self.assertEqual(job.chunking.current_chunk, 1)
        self.assertIn("compute_embed", end_model.calls)
        self.assertEqual(pipe.sent_jobs[2].pass_idx, 2)

    def test_a_bounced_decode_token_is_resent_not_embedded_again(self):
        # A short prompt never chunks, so the second pass is a decode token
        job, pipe = self.make_origin(FakeEndModelContinue())
        self.dispatch(job, pipe, FakeEndModelContinue())
        self.return_from_pipe(job, pipe)
        self.dispatch(job, pipe, FakeEndModelContinue())

        sent = pipe.sent_jobs[1]
        current_token = job.current_token
        input_ids = list(job.input_ids)

        self.assertTrue(self.bounce(job, pipe))

        end_model = FakeEndModelContinue()
        make_processor(job=job, pipe=pipe, end_model=end_model).run()

        self.assertEqual(end_model.calls, [])
        self.assertEqual(job.current_token, current_token)
        self.assertEqual(job.input_ids, input_ids)
        self.assertEqual(len(pipe.sent_jobs), 3)
        self.assertIs(pipe.sent_jobs[2].data, sent.data)
        self.assertEqual(pipe.sent_jobs[2].pass_idx, sent.pass_idx)

    def test_a_bounce_for_a_pass_already_replaced_is_dropped(self):
        job, pipe = self.make_origin(FakeEndModelContinue())
        self.dispatch(job, pipe, FakeEndModelContinue())
        self.return_from_pipe(job, pipe)
        self.dispatch(job, pipe, FakeEndModelContinue())

        self.assertFalse(self.bounce(job, pipe, pass_idx=1))
        self.assertFalse(job.replaying)
        self.assertIsNone(job.receive_error)


if __name__ == "__main__":
    unittest.main()