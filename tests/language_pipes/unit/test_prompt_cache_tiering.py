"""Host tiering (cache_tier_plan.md): an entry survives budget pressure by
moving to host RAM instead of being evicted, and is promoted back onto the
device for whichever job borrows it next.

CI has no GPU, so devices are exercised with stub tensors rather than real
cuda tensors - an object with `.device`, `.to()`, `.numel()` and
`.element_size()`, in the style of `test_prompt_cache_share.py`'s `cache_of`.
A stub's `.to()` returns a new stub rather than mutating in place, which is
exactly the contract `move_cache` relies on for real tensors too.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.prompt_cache import (
    BLOCK_SIZE,
    MIN_CACHE_TOKENS,
    PromptCache,
    copy_cache,
    layer_devices,
    move_cache,
)

CUDA0 = torch.device("cuda:0")
CUDA1 = torch.device("cuda:1")
CPU = torch.device("cpu")


class StubTensor:
    """Just enough of a tensor for `move_cache` and `cache_ram_gb`."""

    def __init__(self, device, tag="t"):
        self.device = torch.device(device)
        self.tag = tag
        self.to_calls = 0

    def to(self, device):
        moved = StubTensor(device, self.tag)
        self.to_calls += 1
        return moved

    def numel(self) -> int:
        return 4

    def element_size(self) -> int:
        return 4


class StubLayer:
    """Only defines the attributes it is given, so `_is_in_place_layer`
    (which keys off `hasattr`) sees a growth-style layer unless told
    otherwise - the same distinction `DynamicLayer` vs `LinearAttentionLayer`
    draws in the real model."""

    def __init__(self, **kwargs):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            if name in kwargs:
                setattr(self, name, kwargs[name])


def cache_of(*layers) -> DynamicCache:
    cache = DynamicCache()
    cache.layers = list(layers)
    cache.layer_class_to_replicate = None
    return cache


def make_prompt_cache(host_tokens: int = 100000) -> PromptCache:
    return PromptCache(lambda: 300, lambda: 100000, lambda: host_tokens)


class MoveCacheTests(unittest.TestCase):
    def test_walks_dicts_and_skips_none(self):
        recurrent = {0: StubTensor(CUDA0), 1: None}
        layer = StubLayer(keys=StubTensor(CUDA0), values=None, recurrent_states=recurrent)
        cache = cache_of(layer)

        move_cache(cache, [CPU])

        self.assertEqual(layer.keys.device, CPU)
        self.assertIsNone(layer.values)
        self.assertEqual(layer.recurrent_states[0].device, CPU)
        self.assertIsNone(layer.recurrent_states[1])

    def test_a_none_device_entry_is_left_alone(self):
        layer = StubLayer(keys=StubTensor(CUDA0))
        cache = cache_of(layer)

        move_cache(cache, [None])

        self.assertEqual(layer.keys.device, CUDA0)

    def test_a_tensor_already_on_the_target_device_is_untouched(self):
        tensor = StubTensor(CPU)
        layer = StubLayer(keys=tensor)
        cache = cache_of(layer)

        move_cache(cache, [CPU])

        self.assertIs(layer.keys, tensor)
        self.assertEqual(tensor.to_calls, 0)

    def test_moves_two_layers_independently(self):
        layer0 = StubLayer(keys=StubTensor(CUDA0))
        layer1 = StubLayer(keys=StubTensor(CUDA1))
        cache = cache_of(layer0, layer1)

        move_cache(cache, [CPU, CPU])

        self.assertEqual(layer0.keys.device, CPU)
        self.assertEqual(layer1.keys.device, CPU)


class CopyCacheDevicesTests(unittest.TestCase):
    def test_devices_moves_only_the_copy(self):
        layer = StubLayer(keys=StubTensor(CPU), values=StubTensor(CPU))
        source = cache_of(layer)

        copied = copy_cache(source, devices=[CUDA0])

        self.assertEqual(copied.layers[0].keys.device, CUDA0)
        self.assertEqual(source.layers[0].keys.device, CPU)

    def test_without_devices_the_copy_keeps_the_source_device(self):
        layer = StubLayer(keys=StubTensor(CUDA0))
        source = cache_of(layer)

        copied = copy_cache(source)

        self.assertEqual(copied.layers[0].keys.device, CUDA0)


class DemoteAndPromoteTests(unittest.TestCase):
    """`PromptCache.demote` / `.adopt` on a device-resident entry - the
    mechanism the design doc's invariant rests on."""

    def store(self, cache: PromptCache, cache_id: bytes, *layers) -> None:
        job_cache = cache_of(*layers)
        cache.store(cache_id, job_cache, "node-a", "model-1", ["proc"], 0, len(layers) - 1, MIN_CACHE_TOKENS)

    def test_demoting_moves_the_entrys_own_tensors_to_the_host(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0), values=StubTensor(CUDA0)))

        self.assertTrue(cache.demote(cache_id))

        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertFalse(entry.on_device)
        self.assertEqual(entry.cache.layers[0].keys.device, CPU)

    def test_a_demoted_entry_is_still_found_by_lookup(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))
        cache.demote(cache_id)

        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)

        self.assertIsNotNone(entry)

    def test_demoting_leaves_an_earlier_borrowers_copy_untouched(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        borrowed = cache.adopt(entry)

        cache.demote(cache_id)

        self.assertEqual(borrowed.layers[0].keys.device, CUDA0)

    def test_demoting_twice_is_a_no_op_the_second_time(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))

        self.assertTrue(cache.demote(cache_id))
        self.assertFalse(cache.demote(cache_id))

    def test_demoting_a_missing_entry_is_harmless(self):
        cache = make_prompt_cache()
        self.assertFalse(cache.demote(b"nope" * 8))

    def test_demoting_with_the_host_tier_disabled_is_a_no_op(self):
        """`get_max_cache_host_tokens` defaults to 0 - every test written
        before tiering existed - so this is also today's exact behavior."""
        cache = PromptCache(lambda: 300, lambda: 100000)
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))

        self.assertFalse(cache.demote(cache_id))
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertTrue(entry.on_device)

    def test_a_cpu_only_node_has_no_tiers(self):
        """Where the entry's own tensors are already on the host, demoting is
        a no-op and it stays counted as device-resident - a CPU node must not
        silently acquire a larger cache than `max_cache_tokens` promises."""
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CPU)))

        self.assertFalse(cache.demote(cache_id))
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertTrue(entry.on_device)

    def test_layer_devices_survive_a_demote_and_promote_round_trip_on_two_devices(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(
            cache, cache_id,
            StubLayer(keys=StubTensor(CUDA0)),
            StubLayer(keys=StubTensor(CUDA1))
        )

        cache.demote(cache_id)
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 1)
        assert entry is not None
        promoted = cache.adopt(entry)

        self.assertEqual(promoted.layers[0].keys.device, CUDA0)
        self.assertEqual(promoted.layers[1].keys.device, CUDA1)
        # The entry itself stays on the host - only the borrower's copy moved.
        self.assertFalse(entry.on_device)
        self.assertEqual(entry.cache.layers[0].keys.device, CPU)
        self.assertEqual(entry.cache.layers[1].keys.device, CPU)

    def test_promoting_counts_a_promotion(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))
        cache.demote(cache_id)
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None

        cache.adopt(entry)

        self.assertEqual(cache.stats().promotions, 1)

    def test_adopting_a_device_resident_entry_promotes_nothing(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        self.store(cache, cache_id, StubLayer(keys=StubTensor(CUDA0)))
        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None

        cache.adopt(entry)

        self.assertEqual(cache.stats().promotions, 0)


class TwoTierBudgetTests(unittest.TestCase):
    def store(self, cache: PromptCache, name: bytes, device=CUDA0):
        cache.store(
            name, cache_of(StubLayer(keys=StubTensor(device))),
            "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS
        )

    def lookup(self, cache: PromptCache, name: bytes):
        return cache.lookup(name, "node-a", "model-1", ["proc"], 0, 0)

    def test_device_pressure_demotes_rather_than_evicting(self):
        cache = PromptCache(lambda: 300, lambda: MIN_CACHE_TOKENS, lambda: 100000)
        self.store(cache, b"a" * 32)

        # A second entry does not fit the device budget alongside the first.
        self.assertTrue(cache.reserve("job-1", MIN_CACHE_TOKENS))

        entry = self.lookup(cache, b"a" * 32)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.on_device)
        self.assertEqual(cache.stats().demotions, 1)
        self.assertEqual(cache.stats().evictions, 0)

    def test_a_host_tier_of_zero_makes_demotion_an_eviction(self):
        """Today's behavior, preserved as the fallback when the host tier is
        switched off."""
        cache = PromptCache(lambda: 300, lambda: MIN_CACHE_TOKENS, lambda: 0)
        self.store(cache, b"a" * 32)

        self.assertTrue(cache.reserve("job-1", MIN_CACHE_TOKENS))

        self.assertIsNone(self.lookup(cache, b"a" * 32))
        self.assertEqual(cache.stats().evictions, 1)
        self.assertEqual(cache.stats().demotions, 0)

    def test_host_pressure_evicts_the_host_lru(self):
        cache = PromptCache(lambda: 300, lambda: MIN_CACHE_TOKENS, lambda: MIN_CACHE_TOKENS)
        self.store(cache, b"a" * 32)
        cache.demote(b"a" * 32)
        self.store(cache, b"b" * 32)

        # Demoting "b" pushes the host tier over its own budget of one entry.
        cache.demote(b"b" * 32)
        with cache._lock:
            cache._evict_host_lru_while_over_budget_locked()

        self.assertIsNone(self.lookup(cache, b"a" * 32))
        self.assertIsNotNone(self.lookup(cache, b"b" * 32))
        self.assertEqual(cache.stats().evictions, 1)

    def test_used_host_tokens_counts_only_host_resident_entries(self):
        cache = PromptCache(lambda: 300, lambda: 100000, lambda: 100000)
        self.store(cache, b"a" * 32)
        self.store(cache, b"b" * 32)
        cache.demote(b"a" * 32)

        self.assertEqual(cache.used_host_tokens(), MIN_CACHE_TOKENS)
        self.assertEqual(cache.used_tokens(), MIN_CACHE_TOKENS)

    def test_stats_split_device_and_host_totals(self):
        cache = PromptCache(lambda: 300, lambda: 100000, lambda: 100000)
        self.store(cache, b"a" * 32)
        self.store(cache, b"b" * 32)
        cache.demote(b"a" * 32)

        stats = cache.stats()

        self.assertEqual(stats.entries, 2)
        self.assertEqual(stats.tokens, MIN_CACHE_TOKENS)
        self.assertEqual(stats.host_tokens, MIN_CACHE_TOKENS)
        self.assertEqual(stats.host_budget, 100000)
        self.assertEqual(stats.demotions, 1)


class DemotionQueueTests(unittest.TestCase):
    """`demote_for_job` / `drain_demotions`: the off-thread path job
    completion uses so a D2H never runs on the job-processing thread."""

    def make_job(self, *touched_ids: bytes):
        class FakeCaching:
            def __init__(self):
                self.touched_ids = list(touched_ids)

        class FakeJob:
            def __init__(self):
                self.caching = FakeCaching()

        return FakeJob()

    def test_demote_for_job_does_not_move_anything_until_drained(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        cache.store(cache_id, cache_of(StubLayer(keys=StubTensor(CUDA0))),
                    "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS)

        cache.demote_for_job(self.make_job(cache_id))

        entry = cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertTrue(entry.on_device)

    def test_draining_demotes_every_touched_entry(self):
        cache = make_prompt_cache()
        for name in (b"a" * 32, b"b" * 32):
            cache.store(name, cache_of(StubLayer(keys=StubTensor(CUDA0))),
                        "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS)
        cache.demote_for_job(self.make_job(b"a" * 32, b"b" * 32))

        changed = cache.drain_demotions()

        self.assertEqual(changed, 2)
        for name in (b"a" * 32, b"b" * 32):
            entry = cache.lookup(name, "node-a", "model-1", ["proc"], 0, 0)
            assert entry is not None
            self.assertFalse(entry.on_device)

    def test_a_job_that_touched_nothing_enqueues_nothing(self):
        cache = make_prompt_cache()

        cache.demote_for_job(self.make_job())

        self.assertEqual(cache.drain_demotions(), 0)

    def test_draining_an_empty_queue_is_a_no_op(self):
        cache = make_prompt_cache()
        self.assertEqual(cache.drain_demotions(), 0)

    def test_draining_twice_is_idempotent(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        cache.store(cache_id, cache_of(StubLayer(keys=StubTensor(CUDA0))),
                    "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS)
        cache.demote_for_job(self.make_job(cache_id))

        first = cache.drain_demotions()
        second = cache.drain_demotions()

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)

    def test_an_entry_evicted_between_enqueue_and_drain_is_skipped(self):
        cache = make_prompt_cache()
        cache_id = b"a" * 32
        cache.store(cache_id, cache_of(StubLayer(keys=StubTensor(CUDA0))),
                    "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS)
        cache.demote_for_job(self.make_job(cache_id))
        with cache._lock:
            del cache._entries[cache_id]

        changed = cache.drain_demotions()

        self.assertEqual(changed, 0)


class SweepDemotionTests(unittest.TestCase):
    def store(self, cache: PromptCache, name: bytes):
        cache.store(name, cache_of(StubLayer(keys=StubTensor(CUDA0))),
                    "node-a", "model-1", ["proc"], 0, 0, MIN_CACHE_TOKENS)

    def test_an_entry_no_live_job_holds_is_demoted(self):
        cache = make_prompt_cache()
        self.store(cache, b"a" * 32)

        cache.sweep(live_ids=set())

        entry = cache.lookup(b"a" * 32, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertFalse(entry.on_device)

    def test_an_entry_a_live_job_still_holds_is_left_alone(self):
        cache = make_prompt_cache()
        self.store(cache, b"a" * 32)

        cache.sweep(live_ids={b"a" * 32})

        entry = cache.lookup(b"a" * 32, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertTrue(entry.on_device)

    def test_no_live_ids_argument_leaves_device_residency_untouched(self):
        """Backward compatible with every caller that predates tiering."""
        cache = make_prompt_cache()
        self.store(cache, b"a" * 32)

        cache.sweep()

        entry = cache.lookup(b"a" * 32, "node-a", "model-1", ["proc"], 0, 0)
        assert entry is not None
        self.assertTrue(entry.on_device)

    def test_sweep_still_expires_by_ttl_alongside_demotion(self):
        cache = make_prompt_cache()
        self.store(cache, b"a" * 32)
        with cache._lock:
            cache._entries[b"a" * 32].expires_at -= 10000

        cache.sweep(live_ids=set())

        self.assertEqual(cache.stats().entries, 0)


if __name__ == "__main__":
    unittest.main()
