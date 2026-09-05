"""Two nodes, two requests, one prefix - the thing Phase 2 exists for.

Everything here runs through the real wire format and the real `JobTracker`
entry point, because the distributed part of the prompt cache is exactly the
part that lives in the packet: the origin plans, tags and stores, and the node
hosting the rest of the pipe has to end up holding the same positions from
nothing but what the packet told it.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

import torch
from transformers import PretrainedConfig

from language_pipes.jobs.cache_policy import CacheOutcome, CachePolicy
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, PromptCache
from language_pipes.util.enums import ComputeStep

from util import (
    FakeEndModel,
    FakeModel,
    PipeWrapper,
    make_job,
    make_job_data,
    make_processor,
    mock_complete,
)

ORIGIN = "node-1"
RELAY = "node-b"
NUM_LAYERS = 2
PROMPT_TOKENS = BLOCK_SIZE * 4 + 5  # 517: four whole blocks and a tail


def append(job, positions: int):
    """Add `positions` entries to this node's cache, the way a real layer does."""
    keys = torch.zeros((1, 1, positions, 1))
    job.cache.update(keys, keys.clone(), 0)


class CachingEndModel(FakeEndModel):
    """The origin: tokenizes a prompt worth caching and records what it embeds."""

    def __init__(self, prompt_length: int = PROMPT_TOKENS, **kwargs):
        super().__init__(**kwargs)
        self.prompt_length = prompt_length
        self.embeds = []

    def tokenize(self, job):
        job.input_ids = list(range(self.prompt_length))
        job.prompt_tokens = self.prompt_length
        job.next_step()

    def compute_embed(self, job, chunk_start=0, chunk_end=-1):  # noqa: ARG002
        past = job.past_seen_tokens()
        end = job.chunking.get_range()[1] if job.current_token == 0 else len(job.input_ids)
        self.embeds.append((past, end))
        job.data = make_job_data()
        job.data.state = torch.zeros((1, 1))
        job.data.cache_position = torch.arange(past, end)
        job.next_step()

    def compute_layers(self, job):
        append(job, len(job.data.cache_position))
        super().compute_layers(job)


class CachingModel(FakeModel):
    """The node that hosts the rest of the pipe and nothing else."""

    def process_job(self, job):
        append(job, len(job.data.cache_position))
        super().process_job(job)


def transport(network_job: NetworkJob) -> NetworkJob:
    """Put the packet through the wire format, the way the router does."""
    restored, valid = NetworkJob.from_bytes(network_job.to_bytes())
    assert valid
    return restored


class TwoNodePipe:
    """An origin holding the end model and layer 0, a relay holding layer 1.

    Each node has its own `PromptCache`, as two processes would.
    """

    def __init__(self, origin_seconds: int = 300, relay_seconds: int = 300):
        self.origin_cache = PromptCache(lambda: origin_seconds, lambda: 100000)
        self.relay_cache = PromptCache(lambda: relay_seconds, lambda: 100000)
        self.end_model = CachingEndModel(num_local_layers=1)
        self.origin_pipe = PipeWrapper(ORIGIN, "model-a", [
            FakeModel(RELAY, 1, 1, virtual=True, num_hidden_layers=NUM_LAYERS)
        ])
        self.relay_pipe = PipeWrapper(RELAY, "model-a", [
            CachingModel(RELAY, 1, 1, virtual=False, num_hidden_layers=NUM_LAYERS)
        ])
        with patch("language_pipes.jobs.job_tracker.Thread"):
            self.relay_tracker = JobTracker(self.relay_cache)
        self.relay_tracker.shutdown = True
        self.origin = None
        self.relay = None
        self.outcome = CacheOutcome.OK

    def start_request(self):
        job = make_job(complete=mock_complete)
        job.origin_node_id = ORIGIN
        job.pipe_id = self.origin_pipe.pipe_id
        job.compute_step = ComputeStep.TOKENIZE
        job.caching.options.enabled = True
        job.caching.scope = self.origin_cache.scope(ORIGIN, "api-key", "tenant")
        self.origin = job
        self.relay = None
        return job

    def run_origin(self) -> NetworkJob:
        make_processor(
            job=self.origin, pipe=self.origin_pipe, end_model=self.end_model,
            node_id=ORIGIN, prompt_cache=self.origin_cache
        ).run()
        return self.origin_pipe.sent_jobs[-1]

    def run_relay(self, packet: NetworkJob):
        restored = transport(packet)
        if self.relay is None:
            policy = CachePolicy(RELAY, self.relay_pipe, None, self.relay_cache)
            self.relay, self.outcome = self.relay_tracker.add_job(
                restored,
                PretrainedConfig(num_hidden_layers=NUM_LAYERS),  # pyright: ignore[reportCallIssue]
                "model-a",
                policy
            )
            if self.relay is None:
                return None
        assert self.relay.receive_network_job(restored, RELAY)
        make_processor(
            job=self.relay, pipe=self.relay_pipe, end_model=None,
            node_id=RELAY, prompt_cache=self.relay_cache
        ).run()
        return self.relay_pipe.sent_jobs[-1]

    def prefill(self):
        """Run one request's prefill all the way through, chunk by chunk."""
        while True:
            returned = self.run_relay(self.run_origin())
            if returned is None:
                return
            self.origin.receive_network_job(transport(returned), ORIGIN)
            if not self.origin.chunking.has_more():
                return

    def request(self):
        job = self.start_request()
        self.prefill()
        return job

    def positions(self):
        return (self.origin.cache.get_seq_length(), self.relay.cache.get_seq_length())


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class DistributedReuseTests(unittest.TestCase):
    def test_the_first_request_stores_the_prompt_boundary_on_both_nodes(self):
        pipe = TwoNodePipe()

        pipe.request()

        self.assertEqual(pipe.origin_cache.stats().entries, 1)
        self.assertEqual(pipe.relay_cache.stats().entries, 1)
        self.assertEqual(pipe.origin_cache.stats().tokens, BLOCK_SIZE * 4)
        self.assertEqual(pipe.relay_cache.stats().tokens, BLOCK_SIZE * 4)

    def test_both_nodes_store_the_prefix_under_the_same_id(self):
        pipe = TwoNodePipe()

        job = pipe.request()

        cache_id = job.caching.ids[4]
        self.assertIn(cache_id, pipe.origin_cache._entries)
        self.assertIn(cache_id, pipe.relay_cache._entries)

    def test_the_second_request_skips_the_prefill_the_first_paid_for(self):
        pipe = TwoNodePipe()
        pipe.request()
        first_embeds = len(pipe.end_model.embeds)
        pipe.end_model.embeds = []

        second = pipe.request()

        self.assertEqual(second.caching.prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(second.caching.cached_tokens, BLOCK_SIZE * 4)
        self.assertLess(len(pipe.end_model.embeds), first_embeds)
        self.assertEqual(pipe.end_model.embeds, [(BLOCK_SIZE * 4, PROMPT_TOKENS)])

    def test_the_relay_adopts_the_same_prefix_the_origin_did(self):
        pipe = TwoNodePipe()
        pipe.request()

        pipe.request()

        self.assertEqual(pipe.relay.caching.prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(pipe.outcome, CacheOutcome.OK)

    def test_both_caches_end_the_second_request_the_same_length(self):
        """The invariant the whole design rests on: a prefix is reusable only
        while every node holds exactly the same positions for it."""
        pipe = TwoNodePipe()
        pipe.request()
        self.assertEqual(pipe.positions(), (PROMPT_TOKENS, PROMPT_TOKENS))

        pipe.request()

        self.assertEqual(pipe.positions(), (PROMPT_TOKENS, PROMPT_TOKENS))

    def test_the_read_tags_ride_only_the_first_pass(self):
        pipe = TwoNodePipe()
        pipe.request()
        pipe.request()

        tagged = [j for j in pipe.origin_pipe.sent_jobs if j.cache_use_id != b""]
        self.assertEqual(len(tagged), 1)

    def test_a_relay_with_caching_off_reports_a_miss_instead_of_computing(self):
        pipe = TwoNodePipe()
        pipe.request()
        # The relay's entries go away - its cache was switched off, or swept.
        pipe.relay_cache.clear()

        pipe.start_request()
        pipe.prefill()

        self.assertEqual(pipe.outcome, CacheOutcome.MISS)
        self.assertIsNone(pipe.relay)
        # The origin adopted, so it must not be left believing the prefix is
        # live on the pipe: that is what the MISS answer is for.
        self.assertEqual(pipe.origin.caching.prefix_len, BLOCK_SIZE * 4)

    def test_a_relay_that_never_cached_leaves_the_second_request_correct(self):
        """`max_cache_time = 0` on one node only: the origin still cannot reuse
        anything, because the relay refuses every pass it cannot adopt."""
        pipe = TwoNodePipe(relay_seconds=0)

        pipe.request()

        self.assertEqual(pipe.origin_cache.stats().entries, 1)
        self.assertEqual(pipe.relay_cache.stats().entries, 0)
        self.assertEqual(pipe.positions(), (PROMPT_TOKENS, PROMPT_TOKENS))


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class GatewayOriginTests(unittest.TestCase):
    """An origin that serves the API and holds the end model but none of the
    layers. It has no slice to snapshot, but it is still the only node that sees
    tokens, so it has to record the boundary or nothing on the pipe can reuse."""

    def setUp(self):
        self.pipe = TwoNodePipe()
        self.pipe.end_model = CachingEndModel(num_local_layers=0)
        self.pipe.origin_pipe = PipeWrapper(ORIGIN, "model-a", [
            FakeModel(RELAY, 0, 1, virtual=True, num_hidden_layers=NUM_LAYERS)
        ])
        self.pipe.relay_pipe = PipeWrapper(RELAY, "model-a", [
            CachingModel(RELAY, 0, 1, virtual=False, num_hidden_layers=NUM_LAYERS)
        ])

    def test_the_origin_records_the_boundary_it_holds_no_state_for(self):
        self.pipe.request()

        self.assertEqual(self.pipe.origin_cache.stats().entries, 1)
        self.assertEqual(self.pipe.relay_cache.stats().entries, 1)

    def test_the_second_request_still_reuses_the_prefix(self):
        self.pipe.request()
        self.pipe.end_model.embeds = []

        second = self.pipe.request()

        self.assertEqual(second.caching.cached_tokens, BLOCK_SIZE * 4)
        self.assertEqual(self.pipe.end_model.embeds, [(BLOCK_SIZE * 4, PROMPT_TOKENS)])
        self.assertEqual(self.pipe.relay.cache.get_seq_length(), PROMPT_TOKENS)


if __name__ == "__main__":
    unittest.main()
