import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

import torch

from language_pipes.jobs.job_processor import JobState
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, PromptCache
from language_pipes.util.enums import ComputeStep

from util import make_processor, make_job, make_job_data, FakeModel, TrackingModel, PipeWrapper

class TestProcessLayersState(unittest.TestCase):
    """Tests for the _state_process_layers method."""

    def test_transitions_to_done_when_local_model_missing(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = make_job_data()

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_send_for_remote_layer(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        remote_model = FakeModel("node-b", 1, 1, virtual=True)
        pipe = PipeWrapper("node-a", "model-a", [local_model, remote_model])

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.SEND)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_process_layers_for_local_segment(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 0, virtual=False, num_hidden_layers=2)
        next_model = FakeModel("node-a", 1, 1, virtual=False)
        pipe = PipeWrapper("node-a", "model-a", [local_model, next_model])

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_head_after_all_layers(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [local_model])

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.HEAD)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_transitions_to_embed_for_prefill(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.prompt_tokens = 24
        job.init_chunking()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [local_model])

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.EMBED)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)

    def test_transitions_to_head_after_prefill(self):
        job = make_job()
        job.origin_node_id = "node-1"
        job.prompt_tokens = 24
        job.init_chunking()
        job.chunking.current_chunk = 5
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.last_update = 0

        local_model = TrackingModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [local_model])

        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.HEAD)
        self.assertTrue(local_model.processed)
        self.assertGreater(job.last_update, 0)


class TestReplayedPass(unittest.TestCase):
    """A node that already computed a pass must forward what it produced when
    the origin sends that pass again. Running the layers a second time would
    append the same keys and values to the cache twice."""

    def incoming(self, job, pass_idx: int) -> NetworkJob:
        data = make_job_data()
        data.shared_kv_states = {"full_attention": (torch.zeros((1, 1)), torch.zeros((1, 1)))}
        return NetworkJob(
            job_id=job.job_id,
            pipe_id=job.pipe_id,
            origin_node_id=job.origin_node_id,
            current_layer=0,
            data=data,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=pass_idx
        )

    def run_pass(self, job, pipe, pass_idx: int):
        self.assertTrue(job.receive_network_job(self.incoming(job, pass_idx), "node-1"))
        make_processor(job=job, pipe=pipe, end_model=None).run()

    def test_a_repeated_pass_is_forwarded_without_running_the_layers(self):
        # The job started on node-a; this node (node-1) only hosts layers.
        job = make_job()
        local_model = TrackingModel("node-1", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [local_model])

        self.run_pass(job, pipe, 1)
        sent = pipe.sent_jobs[0]
        local_model.processed = False

        self.assertTrue(job.receive_network_job(self.incoming(job, 1), "node-1"))
        self.assertTrue(job.passes.replaying)

        processor = make_processor(job=job, pipe=pipe, end_model=None)
        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.SEND])
        self.assertFalse(local_model.processed)
        self.assertEqual(len(pipe.sent_jobs), 2)
        self.assertIs(pipe.sent_jobs[1].data, sent.data)
        self.assertIs(pipe.sent_jobs[1].data.shared_kv_states, sent.data.shared_kv_states)
        self.assertEqual(pipe.sent_jobs[1].pass_idx, 1)

    def test_the_next_pass_runs_the_layers_again(self):
        job = make_job()
        local_model = TrackingModel("node-1", 0, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [local_model])

        self.run_pass(job, pipe, 1)
        local_model.processed = False
        self.run_pass(job, pipe, 2)

        self.assertTrue(local_model.processed)
        self.assertFalse(job.passes.replaying)
        self.assertEqual(pipe.sent_jobs[1].pass_idx, 2)




class LayerNodeWriteTests(unittest.TestCase):
    """A layer node stores its own slice off the tag on the packet - the same
    hook the origin uses, reached through `receive_network_job` rather than
    through a plan it made itself."""

    def setUp(self):
        self.cache = PromptCache(lambda: 300, lambda: 100000)
        self.segment = FakeModel("node-b", 1, 1, num_hidden_layers=2)
        self.pipe = PipeWrapper("node-b", "model-a", [
            FakeModel("node-a", 0, 0, virtual=True, num_hidden_layers=2),
            self.segment
        ])

    def make_job(self):
        return make_job(origin_node_id="node-a", pipe_id=self.pipe.pipe_id)

    def tagged_packet(self, job, tokens: int) -> NetworkJob:
        data = make_job_data()
        data.state = torch.zeros((1, 1))
        data.cache_position = torch.arange(tokens - 32, tokens)
        return NetworkJob(
            job_id=job.job_id,
            pipe_id=job.pipe_id,
            origin_node_id="node-a",
            current_layer=1,
            data=data,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=1,
            cache_write_id=b"\x09" * 32,
            cache_write_tokens=tokens
        )

    def run_pass(self, tokens: int):
        job = self.make_job()
        job.cache.update(torch.ones(1, 1, tokens, 4), torch.ones(1, 1, tokens, 4), 0)
        job.receive_network_job(self.tagged_packet(job, tokens), "node-b")
        processor = make_processor(
            job=job, pipe=self.pipe, end_model=None,
            node_id="node-b", prompt_cache=self.cache
        )
        processor._state_process_layers()
        return job

    def test_the_slice_is_stored_under_the_id_the_origin_named(self):
        self.run_pass(BLOCK_SIZE * 4)

        entry = self.cache.lookup(
            b"\x09" * 32, "node-a", "model-a", [self.segment.process_id], 1, 1
        )
        assert entry is not None
        self.assertEqual(entry.token_count, BLOCK_SIZE * 4)

    def test_the_tag_is_forwarded_so_the_next_node_stores_too(self):
        job = self.run_pass(BLOCK_SIZE * 4)

        self.assertEqual(job.to_network_job().cache_write_id, b"\x09" * 32)

    def test_a_pass_that_does_not_end_where_the_tag_says_is_not_stored(self):
        job = self.make_job()
        job.cache.update(torch.ones(1, 1, 32, 4), torch.ones(1, 1, 32, 4), 0)
        packet = self.tagged_packet(job, BLOCK_SIZE * 4)
        packet.data.cache_position = torch.arange(0, 32)  # pyright: ignore[reportOptionalMemberAccess]
        job.receive_network_job(packet, "node-b")
        processor = make_processor(
            job=job, pipe=self.pipe, end_model=None,
            node_id="node-b", prompt_cache=self.cache
        )

        processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)

    def test_a_node_that_has_more_layers_of_the_pass_to_run_waits(self):
        """Its cache would be short of this pass for every layer of the second
        range, so the snapshot has to wait for the last visit."""
        first = FakeModel("node-b", 1, 1, num_hidden_layers=4)
        later = FakeModel("node-b", 3, 3, num_hidden_layers=4, process_id="proc-late")
        self.pipe = PipeWrapper("node-b", "model-a", [
            first, FakeModel("node-a", 2, 2, virtual=True, num_hidden_layers=4), later
        ])
        job = self.make_job()
        job.cache.update(torch.ones(1, 1, BLOCK_SIZE * 4, 4), torch.ones(1, 1, BLOCK_SIZE * 4, 4), 0)
        packet = self.tagged_packet(job, BLOCK_SIZE * 4)
        job.receive_network_job(packet, "node-b")
        processor = make_processor(
            job=job, pipe=self.pipe, end_model=None,
            node_id="node-b", prompt_cache=self.cache
        )

        processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)
        self.assertEqual(job.caching.pending_write_id, b"\x09" * 32)


if __name__ == "__main__":
    unittest.main()
