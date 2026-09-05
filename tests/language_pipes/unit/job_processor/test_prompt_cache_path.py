"""The origin's read and write path: what `_state_embed`, `_state_process_layers`
and `_state_head` do once a prompt cache is present."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

import torch

from language_pipes.jobs.cache_policy import CachePolicy
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, MIN_CACHE_TOKENS, PromptCache
from language_pipes.util.enums import ComputeStep, JobStatus

from util import (
    FakeEndModel,
    FakeModel,
    PipeWrapper,
    make_job,
    make_job_data,
    make_processor,
)

PROMPT_TOKENS = BLOCK_SIZE * 4 + 5  # 517: a whole number of blocks plus a tail
NUM_LAYERS = 2


class CachingEndModel(FakeEndModel):
    """Tokenizes a prompt long enough to be worth caching and records embeds."""

    def __init__(self, prompt_tokens: int = PROMPT_TOKENS, **kwargs):
        super().__init__(**kwargs)
        self.prompt_tokens = prompt_tokens
        self.embeds = []

    def tokenize(self, job):
        self.calls.append("tokenize")
        job.input_ids = list(range(self.prompt_tokens))
        job.prompt_tokens = self.prompt_tokens
        job.next_step()

    def compute_embed(self, job, chunk_start=0, chunk_end=-1):  # noqa: ARG002
        self.calls.append("compute_embed")
        start, end = job.chunking.get_range()
        self.embeds.append((start, end))
        job.data = make_job_data()
        job.data.state = torch.zeros((1, 1))
        # What a real embed pass records: the absolute positions it covered.
        job.data.cache_position = torch.arange(job.past_seen_tokens(), end)
        job.next_step()


def make_pipe(node_id: str = "node-1", layer_node_id: str | None = None) -> PipeWrapper:
    segment = FakeModel(
        layer_node_id or node_id, 1, NUM_LAYERS - 1, num_hidden_layers=NUM_LAYERS
    )
    return PipeWrapper(node_id, "model-1", [segment])


def make_cache(seconds: int = 300, tokens: int = 100000) -> PromptCache:
    return PromptCache(lambda: seconds, lambda: tokens)


def enable(job, cache: PromptCache, key: str = "tenant-1"):
    job.caching.options.enabled = True
    job.caching.scope = cache.scope(job.origin_node_id, "api-key", key)
    return job


def run_prefill(processor):
    """Drive the origin through the whole prefill, one chunk at a time."""
    processor.run()
    while processor.ctx.job.chunking.has_more():
        processor.ctx.job.compute_step = ComputeStep.EMBED
        processor.state = processor.state.__class__.VALIDATING
        processor.run()


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class EmbedReadPathTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def processor(self, job):
        return make_processor(
            job=job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )

    def seed_entry(self, job, blocks: int):
        """Put an entry in the store for the first `blocks` blocks of the prompt."""
        ids = self.cache.chain(job.caching.scope, list(range(PROMPT_TOKENS)))
        seeded = make_job()
        seeded.cache.update(torch.ones(1, 1, blocks * BLOCK_SIZE, 4),
                            torch.ones(1, 1, blocks * BLOCK_SIZE, 4), 0)
        self.cache.store(
            ids[blocks], seeded.cache, job.origin_node_id, "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1, blocks * BLOCK_SIZE
        )
        return ids[blocks]

    def test_a_hit_starts_the_chunks_after_the_cached_prefix(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        self.seed_entry(job, 3)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, 3 * BLOCK_SIZE)
        self.assertEqual(job.caching.cached_tokens, 3 * BLOCK_SIZE)
        self.assertEqual(self.end_model.embeds[0], (3 * BLOCK_SIZE, 3 * BLOCK_SIZE + 32))

    def test_a_hit_skips_the_cached_chunks_entirely(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        self.seed_entry(job, 3)
        processor = self.processor(job)

        run_prefill(processor)

        # 517 - 384 = 133 tokens left, in chunks of 32.
        self.assertEqual(len(self.end_model.embeds), 5)
        self.assertEqual(self.end_model.embeds[0][0], 3 * BLOCK_SIZE)

    def test_a_miss_prefills_the_whole_prompt(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, 0)
        self.assertEqual(job.caching.cached_tokens, 0)
        self.assertEqual(self.end_model.embeds[0], (0, 32))

    def test_the_whole_prompt_is_never_adopted_leaving_nothing_to_embed(self):
        """An identical request replayed: capped at the largest boundary below
        the prompt, so at least one token is always left to embed."""
        self.end_model = CachingEndModel(
            prompt_tokens=BLOCK_SIZE * 4, num_local_layers=1
        )
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        ids = self.cache.chain(job.caching.scope, list(range(BLOCK_SIZE * 4)))
        for blocks in (3, 4):
            seeded = make_job()
            seeded.cache.update(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), 0)
            self.cache.store(
                ids[blocks], seeded.cache, job.origin_node_id, "model-1",
                [self.end_model.process_id, self.pipe.segments[0].process_id],
                0, NUM_LAYERS - 1, blocks * BLOCK_SIZE
            )

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, 3 * BLOCK_SIZE)

    def test_a_remote_segment_is_told_what_to_adopt(self):
        """The origin's own hit is only half of it: the node hosting the rest of
        the pipe has to adopt the same prefix, so the tags ride the first pass."""
        self.pipe = make_pipe(layer_node_id="node-2")
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        cache_id = self.seed_entry(job, 3)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, 3 * BLOCK_SIZE)
        self.assertEqual(job.caching.use_id, cache_id)
        self.assertEqual(job.caching.use_tokens, 3 * BLOCK_SIZE)
        self.assertGreater(job.caching.reserve_tokens, 0)

        network_job = job.to_network_job()
        self.assertEqual(network_job.cache_use_id, cache_id)
        self.assertEqual(network_job.cache_use_tokens, 3 * BLOCK_SIZE)
        self.assertEqual(network_job.cache_reserve_tokens, job.caching.reserve_tokens)

    def test_a_miss_still_asks_the_other_nodes_to_reserve(self):
        """They have to hold their slice of what this job stores, even though
        there is nothing for them to adopt."""
        self.pipe = make_pipe(layer_node_id="node-2")
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        network_job = job.to_network_job()
        self.assertEqual(network_job.cache_use_id, b"")
        self.assertGreater(network_job.cache_reserve_tokens, 0)

    def test_caching_off_for_the_job_touches_nothing(self):
        job = make_job(origin_node_id="node-1")  # caching disabled
        self.seed_entry(enable(make_job(origin_node_id="node-1"), self.cache), 3)
        before = self.cache.used_tokens()

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.ids, [])
        self.assertFalse(job.caching.reserved)
        # No lookup, no store, no reservation: the store is exactly as it was.
        self.assertEqual(self.cache.used_tokens(), before)
        self.assertEqual(self.cache.stats().hits + self.cache.stats().misses, 0)

    def test_a_short_prompt_is_never_cached(self):
        self.end_model = CachingEndModel(
            prompt_tokens=MIN_CACHE_TOKENS - 1, num_local_layers=1
        )
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.ids, [])
        self.assertEqual(job.caching.write_points, [])

    def test_admission_refusal_leaves_the_job_uncached(self):
        self.cache = make_cache(tokens=BLOCK_SIZE)
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, 0)
        self.assertFalse(job.caching.reserved)
        self.assertEqual(job.caching.ids, [])
        self.assertEqual(job.caching.write_points, [])


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class WritePointTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def processor(self, job):
        return make_processor(
            job=job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )

    def test_the_write_point_is_the_last_block_boundary_in_the_prompt(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.write_points, [BLOCK_SIZE * 4])

    def test_only_the_chunk_that_lands_on_the_boundary_is_tagged(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        processor = self.processor(job)
        tagged = []

        processor._state_embed()
        tagged.append((job.chunking.get_range()[1], job.caching.pending_write_id is not None))
        while job.chunking.has_more():
            job.compute_step = ComputeStep.EMBED
            processor._state_embed()
            tagged.append((job.chunking.get_range()[1], job.caching.pending_write_id is not None))

        marked = [end for end, is_tagged in tagged if is_tagged]
        self.assertEqual(marked, [BLOCK_SIZE * 4])
        self.assertEqual(job.caching.pending_write_tokens, 0)

    def test_a_boundary_already_covered_by_the_adopted_prefix_is_not_rewritten(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        ids = self.cache.chain(job.caching.scope, list(range(PROMPT_TOKENS)))
        seeded = make_job()
        seeded.cache.update(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), 0)
        self.cache.store(
            ids[4], seeded.cache, job.origin_node_id, "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1, BLOCK_SIZE * 4
        )

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(job.caching.write_points, [])


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class StoreOnLayersTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()
        self.job = enable(make_job(origin_node_id="node-1"), self.cache)
        self.processor = make_processor(
            job=self.job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )

    def tag(self, tokens: int):
        self.job.compute_step = ComputeStep.LAYER
        self.job.current_layer = 0
        self.job.caching.pending_write_id = b"\x09" * 32
        self.job.caching.pending_write_tokens = tokens
        self.job.data = make_job_data()
        self.job.data.cache_position = torch.arange(tokens - 32, tokens)
        self.job.cache.update(torch.ones(1, 1, tokens, 4), torch.ones(1, 1, tokens, 4), 0)

    def test_the_slice_is_stored_once_the_pass_finishes_its_last_layer(self):
        self.tag(BLOCK_SIZE * 4)

        self.processor._state_process_layers()

        entry = self.cache.lookup(
            b"\x09" * 32, "node-1", "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1
        )
        assert entry is not None
        self.assertEqual(entry.token_count, BLOCK_SIZE * 4)
        # The tag is not consumed: the packet carrying it goes on to the next
        # node, which stores its own slice under the same ID.
        self.assertEqual(self.job.caching.pending_write_id, b"\x09" * 32)

    def test_a_pass_that_does_not_end_where_the_tag_says_is_not_stored(self):
        self.tag(BLOCK_SIZE * 4)
        # A drifted node: its pass covers fewer positions than the tag claims.
        self.job.data.cache_position = torch.arange(0, 32)

        self.processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)

    def test_an_untagged_pass_stores_nothing(self):
        self.tag(BLOCK_SIZE * 4)
        self.job.caching.pending_write_id = None

        self.processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class DecodeWritePointTests(unittest.TestCase):
    """The entry that makes multi-turn chat hit: prompt plus the answer.

    It cannot be written when the job finishes, because by then no pass is left
    to carry the tag to the other nodes on the pipe - and because the cache
    would cover more positions than the boundary names. So the boundaries the
    answer crosses are tagged as it crosses them, on the one pass whose end
    leaves every node's cache exactly that long.
    """

    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def decoding_job(self, total_tokens: int):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.prompt_tokens = PROMPT_TOKENS
        job.current_token = total_tokens - PROMPT_TOKENS
        job.input_ids = list(range(total_tokens))
        job.caching.ids = self.cache.chain(job.caching.scope, list(range(PROMPT_TOKENS)))
        job.caching.write_points = [BLOCK_SIZE * 4]
        return job

    def policy(self):
        return CachePolicy("node-1", self.pipe, self.end_model, self.cache)

    def test_the_pass_that_lands_on_a_boundary_is_tagged(self):
        job = self.decoding_job(BLOCK_SIZE * 6)

        self.policy().tag_write_point(job)

        ids = self.cache.chain(job.caching.scope, job.input_ids)
        self.assertEqual(job.caching.pending_write_id, ids[6])
        self.assertEqual(job.caching.pending_write_tokens, BLOCK_SIZE * 6)

    def test_a_pass_that_lands_short_of_a_boundary_is_not_tagged(self):
        job = self.decoding_job(BLOCK_SIZE * 6 + 10)

        self.policy().tag_write_point(job)

        self.assertIsNone(job.caching.pending_write_id)
        self.assertEqual(job.caching.write_points, [BLOCK_SIZE * 4])

    def test_each_boundary_the_answer_crosses_is_planned_once(self):
        job = self.decoding_job(BLOCK_SIZE * 6)
        policy = self.policy()

        policy.tag_write_point(job)
        # ... the answer runs on to the next boundary
        job.input_ids = list(range(BLOCK_SIZE * 7))
        policy.tag_write_point(job)

        self.assertEqual(
            job.caching.write_points,
            [BLOCK_SIZE * 4, BLOCK_SIZE * 6, BLOCK_SIZE * 7]
        )
        self.assertEqual(job.caching.pending_write_tokens, BLOCK_SIZE * 7)

    def test_an_uncached_job_is_never_tagged(self):
        job = self.decoding_job(BLOCK_SIZE * 6)
        job.caching.ids = []

        self.policy().tag_write_point(job)

        self.assertIsNone(job.caching.pending_write_id)

    def test_a_node_with_no_budget_stops_every_later_write(self):
        job = self.decoding_job(BLOCK_SIZE * 6)
        job.caching.stop_writing()

        self.policy().tag_write_point(job)

        self.assertEqual(job.caching.write_points, [])
        self.assertIsNone(job.caching.pending_write_id)

    def test_the_stored_entry_is_exactly_as_long_as_it_claims(self):
        """What the boundary names and what the cache holds have to agree: a
        job adopting the entry resumes at the count on the entry, and appends
        onto whatever is actually in it."""
        job = self.decoding_job(BLOCK_SIZE * 6)
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.data.cache_position = torch.tensor([BLOCK_SIZE * 6 - 1])
        job.cache.update(
            torch.ones(1, 1, BLOCK_SIZE * 6, 4), torch.ones(1, 1, BLOCK_SIZE * 6, 4), 0
        )
        policy = self.policy()
        policy.tag_write_point(job)

        policy.store_tagged_pass(job)

        ids = self.cache.chain(job.caching.scope, job.input_ids)
        entry = self.cache.lookup(
            ids[6], "node-1", "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1
        )
        assert entry is not None
        self.assertEqual(entry.token_count, BLOCK_SIZE * 6)
        self.assertEqual(entry.cache.layers[0].keys.shape[-2], BLOCK_SIZE * 6)


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class ExplicitModeTests(unittest.TestCase):
    """In explicit mode the client says where the stable blocks end, so the
    write points come from breakpoints instead of from the end of the prompt -
    and nothing is written for the answer."""

    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def job(self, prefixes, breakpoints=(1,)):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.caching.options.mode = "explicit"
        job.caching.options.breakpoints = list(breakpoints)
        self.end_model.prefixes = prefixes
        return job

    def processor(self, job):
        return make_processor(
            job=job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )

    def test_a_breakpoint_becomes_a_write_point_rounded_to_a_block(self):
        job = self.job([list(range(300))])

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.write_points, [BLOCK_SIZE * 2])
        self.assertEqual(self.end_model.prefix_indices, [1])

    def test_the_implicit_end_of_prompt_boundary_is_not_added(self):
        """The client asked for its boundaries, not ours."""
        job = self.job([list(range(300))])

        self.processor(job)._state_embed()

        self.assertNotIn(BLOCK_SIZE * 4, job.caching.write_points)

    def test_several_breakpoints_are_kept_in_order(self):
        job = self.job([list(range(300)), list(range(500))], breakpoints=(1, 3))

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.write_points, [BLOCK_SIZE * 2, BLOCK_SIZE * 3])

    def test_a_rendering_the_prompt_does_not_start_with_is_dropped(self):
        """A chat template that rewrote an earlier turn would otherwise hand
        back an offset naming a prefix this prompt does not have."""
        job = self.job([[999] + list(range(1, 300))])

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.ids, [])
        self.assertEqual(job.caching.write_points, [])

    def test_a_breakpoint_below_the_minimum_is_dropped(self):
        job = self.job([list(range(MIN_CACHE_TOKENS - 1))])

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.write_points, [])

    def test_no_surviving_breakpoint_runs_the_job_uncached(self):
        job = self.job([])
        before = self.cache.used_tokens()

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.ids, [])
        self.assertFalse(job.caching.reserved)
        self.assertEqual(self.cache.used_tokens(), before)

    def test_a_boundary_the_adopted_prefix_covers_is_not_rewritten(self):
        job = self.job([list(range(300))])
        ids = self.cache.chain(job.caching.scope, list(range(PROMPT_TOKENS)))
        seeded = make_job()
        seeded.cache.update(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), 0)
        self.cache.store(
            ids[2], seeded.cache, job.origin_node_id, "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1, BLOCK_SIZE * 2
        )

        self.processor(job)._state_embed()

        self.assertEqual(job.caching.prefix_len, BLOCK_SIZE * 2)
        self.assertEqual(job.caching.write_points, [])

    def test_the_answer_crossing_a_boundary_is_not_stored(self):
        """Implicit mode keeps writing as the response grows; explicit mode
        writes only where the client marked, so the tail is never stored."""
        job = self.job([list(range(300))])
        job.prompt_tokens = PROMPT_TOKENS
        job.current_token = BLOCK_SIZE
        job.input_ids = list(range(BLOCK_SIZE * 6))
        job.caching.ids = self.cache.chain(job.caching.scope, list(range(PROMPT_TOKENS)))
        job.caching.write_points = [BLOCK_SIZE * 2]

        CachePolicy("node-1", self.pipe, self.end_model, self.cache).tag_write_point(job)

        self.assertEqual(job.caching.write_points, [BLOCK_SIZE * 2])
        self.assertIsNone(job.caching.pending_write_id)


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class CacheWriteTokensTests(unittest.TestCase):
    """`cache_write_tokens` is `cached_tokens`' opposite: what this request had
    to compute and commit, rather than what it got for free."""

    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def policy(self):
        return CachePolicy("node-1", self.pipe, self.end_model, self.cache)

    def stored_job(self, tokens: int, prefix_len: int = 0):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.prompt_tokens = PROMPT_TOKENS
        job.input_ids = list(range(max(tokens, PROMPT_TOKENS)))
        job.caching.ids = self.cache.chain(job.caching.scope, job.input_ids)
        job.caching.prefix_len = prefix_len
        job.caching.cached_tokens = prefix_len
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()
        job.data.cache_position = torch.arange(tokens - 32, tokens)
        job.cache.update(torch.ones(1, 1, tokens, 4), torch.ones(1, 1, tokens, 4), 0)
        return job

    def store(self, job, tokens: int):
        job.caching.pending_write_id = job.caching.ids[tokens // BLOCK_SIZE]
        job.caching.pending_write_tokens = tokens
        self.policy().store_tagged_pass(job)

    def test_a_single_boundary_counts_the_whole_entry(self):
        job = self.stored_job(BLOCK_SIZE * 4)

        self.store(job, BLOCK_SIZE * 4)

        self.assertEqual(job.caching.write_tokens, BLOCK_SIZE * 4)

    def test_an_adopted_prefix_is_not_counted_as_written(self):
        """The client did not pay to compute it, and the entry it came from is
        still in the store."""
        job = self.stored_job(BLOCK_SIZE * 4, prefix_len=BLOCK_SIZE * 3)

        self.store(job, BLOCK_SIZE * 4)

        self.assertEqual(job.caching.write_tokens, BLOCK_SIZE)

    def test_later_boundaries_count_only_what_they_add(self):
        job = self.stored_job(BLOCK_SIZE * 4)
        # The chain grows with the answer; every ID it shares with the prompt's
        # comes out identical, by construction.
        job.input_ids = list(range(BLOCK_SIZE * 5))
        job.caching.ids = self.cache.chain(job.caching.scope, job.input_ids)
        self.store(job, BLOCK_SIZE * 4)

        job.data.cache_position = torch.arange(BLOCK_SIZE * 5 - 32, BLOCK_SIZE * 5)
        job.cache = make_job().cache
        job.cache.update(
            torch.ones(1, 1, BLOCK_SIZE * 5, 4), torch.ones(1, 1, BLOCK_SIZE * 5, 4), 0
        )
        self.store(job, BLOCK_SIZE * 5)

        self.assertEqual(job.caching.write_tokens, BLOCK_SIZE * 5)

    def test_a_skipped_store_counts_nothing(self):
        job = self.stored_job(BLOCK_SIZE * 4)
        # A drifted pass: the tag claims more than the pass covered.
        job.data.cache_position = torch.arange(0, 32)

        self.store(job, BLOCK_SIZE * 4)

        self.assertEqual(job.caching.write_tokens, 0)

    def test_a_layer_node_reports_nothing(self):
        """Only the origin answers a client, and the other nodes store the same
        boundaries under the same IDs anyway."""
        job = self.stored_job(BLOCK_SIZE * 4)
        job.origin_node_id = "node-2"
        job.caching.pending_write_id = job.caching.ids[4]
        job.caching.pending_write_tokens = BLOCK_SIZE * 4

        CachePolicy("node-1", self.pipe, self.end_model, self.cache).store_tagged_pass(job)

        self.assertEqual(job.caching.write_tokens, 0)


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class TwoRequestTests(unittest.TestCase):
    """The loop that actually has to close: one request stores at its prompt
    boundary, the next one with the same prompt adopts it."""

    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def run_request(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        processor = make_processor(
            job=job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )
        run_prefill(processor)
        return job

    def test_the_second_request_skips_the_prefill_the_first_paid_for(self):
        first = self.run_request()
        self.assertEqual(first.caching.prefix_len, 0)
        self.assertEqual(self.cache.stats().entries, 1)
        first_embeds = len(self.end_model.embeds)

        self.end_model.embeds = []
        second = self.run_request()

        self.assertEqual(second.caching.prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(second.caching.cached_tokens, BLOCK_SIZE * 4)
        self.assertLess(len(self.end_model.embeds), first_embeds)
        self.assertEqual(self.end_model.embeds[0][0], BLOCK_SIZE * 4)

    def test_a_different_cache_key_does_not_see_the_entry(self):
        self.run_request()

        other = enable(make_job(origin_node_id="node-1"), self.cache, key="tenant-2")
        processor = make_processor(
            job=other, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )
        run_prefill(processor)

        self.assertEqual(other.caching.prefix_len, 0)

    def test_nothing_is_reused_after_the_hosting_process_is_unloaded(self):
        self.run_request()
        self.cache.clear_process(self.end_model.process_id)

        second = self.run_request()

        self.assertEqual(second.caching.prefix_len, 0)

    def test_a_node_with_caching_switched_off_behaves_as_it_did_before(self):
        self.cache = make_cache(seconds=0)
        first = self.run_request()
        baseline = len(self.end_model.embeds)

        self.end_model.embeds = []
        second = self.run_request()

        self.assertEqual(first.caching.prefix_len, 0)
        self.assertEqual(second.caching.prefix_len, 0)
        self.assertEqual(len(self.end_model.embeds), baseline)
        self.assertEqual(self.cache.stats().entries, 0)
        self.assertEqual(self.cache.used_tokens(), 0)


class LogFieldTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_cache()
        self.pipe = make_pipe()

    def fields(self, job):
        policy = CachePolicy("node-1", self.pipe, FakeEndModel(), self.cache)
        return policy.log_fields(job)

    def test_a_hit_reports_the_scope_prefix_and_never_the_cache_key(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache, key="secret-key")
        job.caching.ids = [job.caching.scope, b"a" * 32]
        job.caching.cached_tokens = 384

        fields = self.fields(job)

        self.assertIn("cache=hit", fields)
        self.assertIn("cached=384", fields)
        self.assertIn(job.caching.scope[:4].hex(), fields)
        self.assertNotIn("secret-key", fields)

    def test_a_miss_is_reported_as_a_miss(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.caching.ids = [job.caching.scope, b"a" * 32]

        self.assertIn("cache=miss", self.fields(job))

    def test_a_job_with_caching_disabled_reports_off(self):
        self.assertIn("cache=off", self.fields(make_job(origin_node_id="node-1")))

    def test_a_job_that_never_reached_the_store_reports_off_not_miss(self):
        """A short prompt, a remote segment or a refused admission all leave the
        job without a chain; none of them is a cache miss."""
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.assertIn("cache=off", self.fields(job))


if __name__ == "__main__":
    unittest.main()
