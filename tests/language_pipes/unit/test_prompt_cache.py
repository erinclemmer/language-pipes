import os
import sys
import unittest
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.prompt_cache import (
    BLOCK_SIZE,
    MIN_CACHE_TOKENS,
    PromptCache,
    chain,
    new_secret,
    scope,
)


def make_cache(layers: int = 2, tokens: int = 4) -> DynamicCache:
    cache = DynamicCache(config=PretrainedConfig(num_hidden_layers=layers))
    for idx in range(layers):
        cache.update(torch.ones(1, 2, tokens, 8), torch.ones(1, 2, tokens, 8), idx)
    return cache


def make_prompt_cache(seconds: int = 300, tokens: int = 100000) -> PromptCache:
    return PromptCache(lambda: seconds, lambda: tokens)


IDENTITY = dict(
    origin_node_id="node-a",
    model_id="model-1",
    process_ids=["proc-end", "proc-layers"],
    start_layer=0,
    end_layer=1,
)


class ChainTests(unittest.TestCase):
    """The chain is what makes two requests recognize a shared prefix."""

    def setUp(self):
        self.secret = new_secret()
        self.root = scope(self.secret, "node-a", "key", "")

    def test_same_tokens_under_one_secret_give_the_same_ids(self):
        tokens = list(range(BLOCK_SIZE * 3))
        self.assertEqual(
            chain(self.secret, self.root, tokens),
            chain(self.secret, self.root, tokens)
        )

    def test_ids_name_whole_block_prefixes(self):
        ids = chain(self.secret, self.root, list(range(BLOCK_SIZE * 3 + 7)))
        # Index 0 is the empty prefix, then one per completed block.
        self.assertEqual(len(ids), 4)
        self.assertEqual(ids[0], self.root)

    def test_a_shared_prefix_shares_its_leading_ids(self):
        base = list(range(BLOCK_SIZE * 3))
        other = base[:BLOCK_SIZE * 2] + [9] * BLOCK_SIZE

        a = chain(self.secret, self.root, base)
        b = chain(self.secret, self.root, other)

        self.assertEqual(a[:3], b[:3])
        self.assertNotEqual(a[3], b[3])

    def test_one_different_token_diverges_from_that_block_on(self):
        base = list(range(BLOCK_SIZE * 2))
        changed = list(base)
        changed[0] = 999

        a = chain(self.secret, self.root, base)
        b = chain(self.secret, self.root, changed)

        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertNotEqual(a[2], b[2])

    def test_scope_separates_api_keys_cache_keys_and_origins(self):
        base = scope(self.secret, "node-a", "key-1", "")
        self.assertNotEqual(base, scope(self.secret, "node-a", "key-2", ""))
        self.assertNotEqual(base, scope(self.secret, "node-a", "key-1", "tenant"))
        self.assertNotEqual(base, scope(self.secret, "node-b", "key-1", ""))

    def test_a_fresh_secret_yields_different_ids_for_the_same_tokens(self):
        """Entries cannot survive a restart, because their IDs cannot be named
        again."""
        tokens = list(range(BLOCK_SIZE))
        other = new_secret()
        self.assertNotEqual(
            chain(self.secret, self.root, tokens),
            chain(other, scope(other, "node-a", "key", ""), tokens)
        )

    def test_fields_are_framed_so_a_split_cannot_be_moved(self):
        """`api_key="a"` + key `"bc"` must not land in the same scope as
        `api_key="ab"` + key `"c"`; with plain concatenation it would."""
        self.assertNotEqual(
            scope(self.secret, "node-a", "a", "bc"),
            scope(self.secret, "node-a", "ab", "c")
        )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_prompt_cache()
        self.cache_id = b"\x01" * 32

    def store(self, **overrides):
        args = dict(IDENTITY)
        args.update(overrides)
        return self.cache.store(
            self.cache_id, make_cache(), args["origin_node_id"], args["model_id"],
            args["process_ids"], args["start_layer"], args["end_layer"],
            args.get("token_count", MIN_CACHE_TOKENS)
        )

    def test_a_stored_entry_is_found_again(self):
        self.assertTrue(self.store())
        entry = self.cache.lookup(self.cache_id, **IDENTITY)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token_count, MIN_CACHE_TOKENS)

    def test_a_prefix_below_the_minimum_is_never_stored(self):
        self.assertFalse(self.store(token_count=BLOCK_SIZE))
        self.assertIsNone(self.cache.lookup(self.cache_id, **IDENTITY))

    def test_another_origin_cannot_reuse_the_entry(self):
        """An ID rides the packet on a distributed pipe, so it must not be a
        bearer token for state it names."""
        self.store()
        identity = dict(IDENTITY, origin_node_id="node-b")
        self.assertIsNone(self.cache.lookup(self.cache_id, **identity))

    def test_a_reloaded_model_process_invalidates_the_entry(self):
        self.store()
        identity = dict(IDENTITY, process_ids=["proc-end", "proc-layers-2"])
        self.assertIsNone(self.cache.lookup(self.cache_id, **identity))

    def test_a_different_layer_range_is_a_miss(self):
        self.store()
        self.assertIsNone(self.cache.lookup(self.cache_id, **dict(IDENTITY, end_layer=3)))

    def test_a_different_model_is_a_miss(self):
        self.store()
        self.assertIsNone(self.cache.lookup(self.cache_id, **dict(IDENTITY, model_id="model-2")))

    def test_adopt_returns_a_cache_the_entry_does_not_share_a_container_with(self):
        self.store()
        entry = self.cache.lookup(self.cache_id, **IDENTITY)
        adopted = self.cache.adopt(entry)

        self.assertIsNot(adopted, entry.cache)
        self.assertEqual(adopted.get_seq_length(), entry.cache.get_seq_length())


class ExpiryTests(unittest.TestCase):
    def setUp(self):
        self.cache_id = b"\x02" * 32

    def store(self, cache: PromptCache):
        cache.store(
            self.cache_id, make_cache(), IDENTITY["origin_node_id"], IDENTITY["model_id"],
            IDENTITY["process_ids"], 0, 1, MIN_CACHE_TOKENS
        )

    def test_an_expired_entry_is_a_miss_and_is_swept(self):
        cache = make_prompt_cache(seconds=300)
        self.store(cache)
        # Reach in rather than sleep: the clock is the thing under test.
        cache._entries[self.cache_id].expires_at = time() - 1

        self.assertIsNone(cache.lookup(self.cache_id, **IDENTITY))
        cache.sweep()
        self.assertEqual(cache.stats().entries, 0)

    def test_reuse_pushes_the_expiry_out(self):
        """Lifetime is measured from last use: a busy prefix stays warm."""
        cache = make_prompt_cache(seconds=300)
        self.store(cache)
        cache._entries[self.cache_id].expires_at = time() + 1

        cache.lookup(self.cache_id, **IDENTITY)

        self.assertGreater(cache._entries[self.cache_id].expires_at, time() + 200)

    def test_a_requested_ttl_is_clamped_to_the_node_limit(self):
        cache = make_prompt_cache(seconds=60)
        cache.store(
            self.cache_id, make_cache(), IDENTITY["origin_node_id"], IDENTITY["model_id"],
            IDENTITY["process_ids"], 0, 1, MIN_CACHE_TOKENS, ttl=86400
        )
        entry = cache._entries[self.cache_id]
        self.assertLessEqual(entry.expires_at - entry.created, 60)


class DisabledTests(unittest.TestCase):
    """Either limit at 0 turns the whole feature off: no reads, no writes, no
    reservations, no memory held."""

    def _assert_inert(self, cache: PromptCache):
        cache_id = b"\x03" * 32
        self.assertFalse(cache.store(
            cache_id, make_cache(), "node-a", "model-1", ["p"], 0, 1, MIN_CACHE_TOKENS
        ))
        self.assertIsNone(cache.lookup(cache_id, **IDENTITY))
        self.assertIsNone(cache.find_longest([b"", cache_id], 1, **IDENTITY))
        self.assertFalse(cache.reserve("job-1", 10))
        self.assertEqual(cache.used_tokens(), 0)
        self.assertEqual(cache.stats().entries, 0)

    def test_zero_max_cache_time_disables_everything(self):
        self._assert_inert(make_prompt_cache(seconds=0))

    def test_zero_max_cache_tokens_disables_everything(self):
        self._assert_inert(make_prompt_cache(tokens=0))


class FindLongestTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_prompt_cache()
        self.ids = [bytes([i]) * 32 for i in range(6)]

    def store_at(self, blocks: int):
        self.cache.store(
            self.ids[blocks], make_cache(), IDENTITY["origin_node_id"],
            IDENTITY["model_id"], IDENTITY["process_ids"], 0, 1, blocks * BLOCK_SIZE
        )

    def test_takes_the_longest_stored_prefix_at_or_below_the_cap(self):
        self.store_at(2)
        self.store_at(4)

        found = self.cache.find_longest(self.ids, 3, **IDENTITY)

        self.assertIsNotNone(found)
        self.assertEqual(found[0], 2)

    def test_a_miss_is_counted_once_not_once_per_candidate(self):
        self.assertIsNone(self.cache.find_longest(self.ids, 5, **IDENTITY))
        stats = self.cache.stats()
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 0)

    def test_never_looks_below_the_minimum_cacheable_prefix(self):
        self.store_at(1)  # 128 tokens, under MIN_CACHE_TOKENS
        self.assertIsNone(self.cache.find_longest(self.ids, 1, **IDENTITY))


class ClearTests(unittest.TestCase):
    def setUp(self):
        self.cache = make_prompt_cache()
        self.a = b"\x0a" * 32
        self.b = b"\x0b" * 32
        self.cache.store(self.a, make_cache(), "node-a", "model-1", ["proc-1"], 0, 1, MIN_CACHE_TOKENS)
        self.cache.store(self.b, make_cache(), "node-a", "model-1", ["proc-2"], 0, 1, MIN_CACHE_TOKENS)

    def test_clear_process_drops_only_that_process_entries(self):
        self.cache.clear_process("proc-1")

        self.assertEqual(self.cache.stats().entries, 1)
        self.assertIsNotNone(self.cache.lookup(
            self.b, "node-a", "model-1", ["proc-2"], 0, 1
        ))

    def test_clear_drops_everything(self):
        self.cache.reserve("job-1", 10)
        self.cache.clear()

        self.assertEqual(self.cache.stats().entries, 0)
        self.assertEqual(self.cache.used_tokens(), 0)


if __name__ == "__main__":
    unittest.main()
