import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.prompt_cache import BLOCK_SIZE, PromptCache


def make_cache() -> DynamicCache:
    cache = DynamicCache(config=PretrainedConfig(num_hidden_layers=1))
    cache.update(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4), 0)
    return cache


class BudgetTests(unittest.TestCase):
    """The budget is counted in tokens because a token count is knowable before
    the KV state exists, which is what admission control needs."""

    def setUp(self):
        self.budget = 10 * BLOCK_SIZE
        self.cache = PromptCache(lambda: 300, lambda: self.budget)

    def store(self, name: bytes, blocks: int):
        self.cache.store(
            name, make_cache(), "node-a", "model-1", ["proc"], 0, 0, blocks * BLOCK_SIZE
        )

    def lookup(self, name: bytes):
        return self.cache.lookup(name, "node-a", "model-1", ["proc"], 0, 0)

    def test_used_tokens_counts_entries_plus_live_reservations(self):
        self.store(b"a" * 32, 2)
        self.cache.reserve("job-1", 100)

        self.assertEqual(self.cache.used_tokens(), 2 * BLOCK_SIZE + 100)

    def test_admission_evicts_least_recently_used_until_the_estimate_fits(self):
        self.store(b"a" * 32, 4)
        self.store(b"b" * 32, 4)
        # Touching `a` makes `b` the oldest, so `b` is what gets dropped.
        self.lookup(b"a" * 32)

        self.assertTrue(self.cache.reserve("job-1", 6 * BLOCK_SIZE))

        self.assertIsNotNone(self.lookup(b"a" * 32))
        self.assertIsNone(self.lookup(b"b" * 32))
        self.assertEqual(self.cache.stats().evictions, 1)

    def test_an_estimate_larger_than_the_budget_evicts_nothing_and_refuses(self):
        self.store(b"a" * 32, 2)

        self.assertFalse(self.cache.reserve("job-1", self.budget + 1))

        self.assertIsNotNone(self.lookup(b"a" * 32))
        self.assertEqual(self.cache.stats().evictions, 0)
        self.assertEqual(self.cache.stats().no_store, 1)

    def test_a_refused_job_holds_no_reservation(self):
        self.assertFalse(self.cache.reserve("job-1", self.budget + 1))
        self.assertEqual(self.cache.used_tokens(), 0)

    def test_release_gives_the_budget_back(self):
        self.cache.reserve("job-1", 500)
        self.cache.release("job-1")

        self.assertEqual(self.cache.used_tokens(), 0)

    def test_releasing_a_job_that_never_reserved_is_harmless(self):
        self.cache.release("job-unknown")
        self.assertEqual(self.cache.used_tokens(), 0)

    def test_reserving_twice_for_one_job_charges_once(self):
        self.assertTrue(self.cache.reserve("job-1", 100))
        self.assertTrue(self.cache.reserve("job-1", 100))

        self.assertEqual(self.cache.used_tokens(), 100)

    def test_a_reused_entry_moves_to_the_back_of_the_eviction_order(self):
        self.store(b"a" * 32, 2)
        self.store(b"b" * 32, 2)
        self.lookup(b"a" * 32)

        self.cache.evict_lru()

        self.assertIsNotNone(self.lookup(b"a" * 32))
        self.assertIsNone(self.lookup(b"b" * 32))

    def test_evicting_cannot_break_a_job_already_running_on_the_entry(self):
        """Adoption handed the job a container of its own; refcounting keeps the
        tensors alive for as long as it holds them."""
        self.store(b"a" * 32, 2)
        entry = self.lookup(b"a" * 32)
        assert entry is not None
        adopted = self.cache.adopt(entry)
        length = adopted.get_seq_length()

        self.cache.evict_lru()

        self.assertEqual(adopted.get_seq_length(), length)
        self.assertIsNone(self.lookup(b"a" * 32))

    def test_evict_on_an_empty_store_reports_nothing_dropped(self):
        self.assertFalse(self.cache.evict_lru())

    def test_stats_report_the_budget_and_the_hit_rate(self):
        self.store(b"a" * 32, 2)
        # ids[2] names the 2-block prefix, which is what was stored.
        self.cache.find_longest(
            [b"", b"x" * 32, b"a" * 32], 2, "node-a", "model-1", ["proc"], 0, 0
        )
        self.cache.find_longest(
            [b"", b"y" * 32, b"z" * 32], 2, "node-a", "model-1", ["proc"], 0, 0
        )

        stats = self.cache.stats()
        self.assertEqual(stats.budget, self.budget)
        self.assertEqual(stats.tokens, 2 * BLOCK_SIZE)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_rate(), 0.5)


class DisabledBudgetTests(unittest.TestCase):
    def test_a_zero_budget_short_circuits_reserve(self):
        cache = PromptCache(lambda: 300, lambda: 0)

        self.assertFalse(cache.reserve("job-1", 1))
        self.assertEqual(cache.used_tokens(), 0)
        self.assertEqual(cache.stats().no_store, 0)


if __name__ == "__main__":
    unittest.main()
