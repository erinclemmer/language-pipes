"""The origin's read and write path: what `_state_embed`, `_state_process_layers`
and `_state_head` do once a prompt cache is present."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

import torch

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
    job.cache_options.enabled = True
    job.cache_scope = cache.scope(job.origin_node_id, "api-key", key)
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
        ids = self.cache.chain(job.cache_scope, list(range(PROMPT_TOKENS)))
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

        self.assertEqual(job.cached_prefix_len, 3 * BLOCK_SIZE)
        self.assertEqual(job.cached_tokens, 3 * BLOCK_SIZE)
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

        self.assertEqual(job.cached_prefix_len, 0)
        self.assertEqual(job.cached_tokens, 0)
        self.assertEqual(self.end_model.embeds[0], (0, 32))

    def test_the_whole_prompt_is_never_adopted_leaving_nothing_to_embed(self):
        """An identical request replayed: capped at the largest boundary below
        the prompt, so at least one token is always left to embed."""
        self.end_model = CachingEndModel(
            prompt_tokens=BLOCK_SIZE * 4, num_local_layers=1
        )
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        ids = self.cache.chain(job.cache_scope, list(range(BLOCK_SIZE * 4)))
        for blocks in (3, 4):
            seeded = make_job()
            seeded.cache.update(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), 0)
            self.cache.store(
                ids[blocks], seeded.cache, job.origin_node_id, "model-1",
                [self.end_model.process_id, self.pipe.segments[0].process_id],
                0, NUM_LAYERS - 1, blocks * BLOCK_SIZE
            )

        self.processor(job)._state_embed()

        self.assertEqual(job.cached_prefix_len, 3 * BLOCK_SIZE)

    def test_no_reuse_when_the_pipe_has_a_remote_segment(self):
        self.pipe = make_pipe(layer_node_id="node-2")
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        self.seed_entry(job, 3)

        self.processor(job)._state_embed()

        self.assertEqual(job.cached_prefix_len, 0)
        self.assertEqual(job.cache_ids, [])

    def test_caching_off_for_the_job_touches_nothing(self):
        job = make_job(origin_node_id="node-1")  # cache_options disabled
        self.seed_entry(enable(make_job(origin_node_id="node-1"), self.cache), 3)
        before = self.cache.used_tokens()

        self.processor(job)._state_embed()

        self.assertEqual(job.cache_ids, [])
        self.assertFalse(job.cache_reserved)
        # No lookup, no store, no reservation: the store is exactly as it was.
        self.assertEqual(self.cache.used_tokens(), before)
        self.assertEqual(self.cache.stats().hits + self.cache.stats().misses, 0)

    def test_a_short_prompt_is_never_cached(self):
        self.end_model = CachingEndModel(
            prompt_tokens=MIN_CACHE_TOKENS - 1, num_local_layers=1
        )
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.cache_ids, [])
        self.assertEqual(job.cache_write_points, [])

    def test_admission_refusal_leaves_the_job_uncached(self):
        self.cache = make_cache(tokens=BLOCK_SIZE)
        job = enable(make_job(origin_node_id="node-1"), self.cache)

        self.processor(job)._state_embed()

        self.assertEqual(job.cached_prefix_len, 0)
        self.assertFalse(job.cache_reserved)
        self.assertEqual(job.cache_ids, [])
        self.assertEqual(job.cache_write_points, [])


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

        self.assertEqual(job.cache_write_points, [BLOCK_SIZE * 4])

    def test_only_the_chunk_that_lands_on_the_boundary_is_tagged(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        processor = self.processor(job)
        tagged = []

        processor._state_embed()
        tagged.append((job.chunking.get_range()[1], job.pending_write_id is not None))
        while job.chunking.has_more():
            job.compute_step = ComputeStep.EMBED
            processor._state_embed()
            tagged.append((job.chunking.get_range()[1], job.pending_write_id is not None))

        marked = [end for end, is_tagged in tagged if is_tagged]
        self.assertEqual(marked, [BLOCK_SIZE * 4])
        self.assertEqual(job.pending_write_tokens, 0)

    def test_a_boundary_already_covered_by_the_adopted_prefix_is_not_rewritten(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        ids = self.cache.chain(job.cache_scope, list(range(PROMPT_TOKENS)))
        seeded = make_job()
        seeded.cache.update(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), 0)
        self.cache.store(
            ids[4], seeded.cache, job.origin_node_id, "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1, BLOCK_SIZE * 4
        )

        self.processor(job)._state_embed()

        self.assertEqual(job.cached_prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(job.cache_write_points, [])


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
        self.job.pending_write_id = b"\x09" * 32
        self.job.pending_write_tokens = tokens
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
        self.assertIsNone(self.job.pending_write_id)

    def test_a_pass_that_does_not_end_where_the_tag_says_is_not_stored(self):
        self.tag(BLOCK_SIZE * 4)
        # A drifted node: its pass covers fewer positions than the tag claims.
        self.job.data.cache_position = torch.arange(0, 32)

        self.processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)

    def test_an_untagged_pass_stores_nothing(self):
        self.tag(BLOCK_SIZE * 4)
        self.job.pending_write_id = None

        self.processor._state_process_layers()

        self.assertEqual(self.cache.stats().entries, 0)


@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class EndOfResponseTests(unittest.TestCase):
    """The entry that makes multi-turn chat hit: prompt plus the answer."""

    def setUp(self):
        self.cache = make_cache()
        self.end_model = CachingEndModel(num_local_layers=1)
        self.pipe = make_pipe()

    def finished_job(self, total_tokens: int):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.prompt_tokens = PROMPT_TOKENS
        job.current_token = total_tokens - PROMPT_TOKENS
        job.input_ids = list(range(total_tokens))
        job.cache_ids = self.cache.chain(job.cache_scope, list(range(PROMPT_TOKENS)))
        job.cache_write_points = [BLOCK_SIZE * 4]
        job.compute_step = ComputeStep.HEAD
        job.status = JobStatus.COMPLETED
        job.data = make_job_data()
        job.data.state = torch.zeros((1, 1))
        job.data.cache_position = torch.tensor([total_tokens - 2])
        job.cache.update(
            torch.ones(1, 1, total_tokens - 1, 4), torch.ones(1, 1, total_tokens - 1, 4), 0
        )
        return job

    def processor(self, job):
        return make_processor(
            job=job, pipe=self.pipe, end_model=self.end_model,
            node_id="node-1", prompt_cache=self.cache
        )

    def test_stores_at_the_largest_boundary_the_cache_actually_covers(self):
        # 6 blocks + 10 tokens generated; the cache covers total - 1 positions.
        job = self.finished_job(BLOCK_SIZE * 6 + 10)

        self.processor(job)._store_end_of_response(job)

        ids = self.cache.chain(job.cache_scope, job.input_ids)
        entry = self.cache.lookup(
            ids[6], "node-1", "model-1",
            [self.end_model.process_id, self.pipe.segments[0].process_id],
            0, NUM_LAYERS - 1
        )
        assert entry is not None
        self.assertEqual(entry.token_count, BLOCK_SIZE * 6)

    def test_nothing_is_stored_when_the_answer_adds_no_whole_block(self):
        job = self.finished_job(BLOCK_SIZE * 4 + 20)

        self.processor(job)._store_end_of_response(job)

        self.assertEqual(self.cache.stats().entries, 0)

    def test_an_uncached_job_stores_nothing_at_completion(self):
        job = self.finished_job(BLOCK_SIZE * 6 + 10)
        job.cache_ids = []

        self.processor(job)._store_end_of_response(job)

        self.assertEqual(self.cache.stats().entries, 0)


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
        self.assertEqual(first.cached_prefix_len, 0)
        self.assertEqual(self.cache.stats().entries, 1)
        first_embeds = len(self.end_model.embeds)

        self.end_model.embeds = []
        second = self.run_request()

        self.assertEqual(second.cached_prefix_len, BLOCK_SIZE * 4)
        self.assertEqual(second.cached_tokens, BLOCK_SIZE * 4)
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

        self.assertEqual(other.cached_prefix_len, 0)

    def test_nothing_is_reused_after_the_hosting_process_is_unloaded(self):
        self.run_request()
        self.cache.clear_process(self.end_model.process_id)

        second = self.run_request()

        self.assertEqual(second.cached_prefix_len, 0)

    def test_a_node_with_caching_switched_off_behaves_as_it_did_before(self):
        self.cache = make_cache(seconds=0)
        first = self.run_request()
        baseline = len(self.end_model.embeds)

        self.end_model.embeds = []
        second = self.run_request()

        self.assertEqual(first.cached_prefix_len, 0)
        self.assertEqual(second.cached_prefix_len, 0)
        self.assertEqual(len(self.end_model.embeds), baseline)
        self.assertEqual(self.cache.stats().entries, 0)
        self.assertEqual(self.cache.used_tokens(), 0)


class LogFieldTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_cache()
        self.pipe = make_pipe()

    def fields(self, job):
        processor = make_processor(
            job=job, pipe=self.pipe, end_model=FakeEndModel(),
            node_id="node-1", prompt_cache=self.cache
        )
        return processor._cache_log_fields(job)

    def test_a_hit_reports_the_scope_prefix_and_never_the_cache_key(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache, key="secret-key")
        job.cache_ids = [job.cache_scope, b"a" * 32]
        job.cached_tokens = 384

        fields = self.fields(job)

        self.assertIn("cache=hit", fields)
        self.assertIn("cached=384", fields)
        self.assertIn(job.cache_scope[:4].hex(), fields)
        self.assertNotIn("secret-key", fields)

    def test_a_miss_is_reported_as_a_miss(self):
        job = enable(make_job(origin_node_id="node-1"), self.cache)
        job.cache_ids = [job.cache_scope, b"a" * 32]

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
