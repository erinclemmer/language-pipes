"""The invariant the whole design rests on: adopting an entry costs no copy,
and a job that appends to what it adopted cannot change the entry.

This holds because `DynamicLayer.update` rebinds `self.keys` to a fresh
`torch.cat` result rather than writing into the old tensor - the docstring's
"in-place" is wrong. Recurrent layers really do write in place and must be
cloned instead. Both facts are transformers internals, so they are asserted here
rather than trusted: an upgrade that starts mutating shared tensors would
silently corrupt cached state, and this file is what fails first.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import (
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
    LinearAttentionLayer,
)

from language_pipes.jobs.prompt_cache import MIN_CACHE_TOKENS, PromptCache, copy_cache


def cache_of(*layers) -> DynamicCache:
    cache = DynamicCache()
    cache.layers = list(layers)
    cache.layer_class_to_replicate = None
    return cache


def kv(tokens: int):
    return torch.arange(tokens * 8, dtype=torch.float32).reshape(1, 1, tokens, 8)


class AppendingLayerTests(unittest.TestCase):
    """Attention layers grow by rebinding, so sharing their tensors is safe."""

    def _assert_append_leaves_the_entry_alone(self, layer):
        entry_cache = cache_of(layer)
        entry_cache.update(kv(4), kv(4), 0)
        frozen_keys = entry_cache.layers[0].keys
        frozen_values = entry_cache.layers[0].values
        before = frozen_keys.clone()
        before_ptr = frozen_keys.data_ptr()

        adopted = copy_cache(entry_cache)
        # The borrower starts out sharing, not copying.
        self.assertEqual(adopted.layers[0].keys.data_ptr(), before_ptr)

        adopted.update(kv(2), kv(2), 0)

        self.assertEqual(frozen_keys.data_ptr(), before_ptr)
        self.assertTrue(torch.equal(frozen_keys, before))
        self.assertTrue(torch.equal(frozen_values, before))
        self.assertIs(entry_cache.layers[0].keys, frozen_keys)

    def test_dynamic_layer_append_does_not_touch_the_entry(self):
        self._assert_append_leaves_the_entry_alone(DynamicLayer())

    def test_sliding_window_layer_append_does_not_touch_the_entry(self):
        self._assert_append_leaves_the_entry_alone(
            DynamicSlidingWindowLayer(sliding_window=1024)
        )

    def test_the_entry_keeps_the_length_it_was_taken_at(self):
        entry_cache = cache_of(DynamicLayer())
        entry_cache.update(kv(4), kv(4), 0)

        snapshot = copy_cache(entry_cache)
        entry_cache.update(kv(3), kv(3), 0)

        self.assertEqual(snapshot.get_seq_length(), 4)
        self.assertEqual(entry_cache.get_seq_length(), 7)

    def test_sliding_window_bookkeeping_is_carried_over(self):
        """`cumulative_length` is what the mask sizing reads; a snapshot that
        lost it would build the wrong mask on resume."""
        entry_cache = cache_of(DynamicSlidingWindowLayer(sliding_window=1024))
        entry_cache.update(kv(6), kv(6), 0)

        snapshot = copy_cache(entry_cache)

        self.assertEqual(snapshot.get_seq_length(), 6)
        entry_cache.update(kv(2), kv(2), 0)
        self.assertEqual(snapshot.get_seq_length(), 6)


class InPlaceLayerTests(unittest.TestCase):
    """Recurrent layers write into a fixed buffer, so they must be copied.

    Their states live in per-state-index dicts, which a shallow copy would share
    outright - so the bookkeeping has to be copied along with the tensors.
    """

    def make_layer(self) -> LinearAttentionLayer:
        layer = LinearAttentionLayer()
        layer.update_conv_state(torch.zeros(1, 2, 4))
        layer.update_recurrent_state(torch.ones(1, 2, 4))
        return layer

    def recurrent(self, cache):
        return cache.layers[0].recurrent_states[0]

    def conv(self, cache):
        return cache.layers[0].conv_states[0]

    def test_the_recurrent_state_is_copied_not_shared(self):
        entry_cache = cache_of(self.make_layer())
        entry_state = self.recurrent(entry_cache)

        adopted = copy_cache(entry_cache)

        self.assertNotEqual(self.recurrent(adopted).data_ptr(), entry_state.data_ptr())

    def test_a_borrowing_job_cannot_corrupt_the_entry(self):
        entry_cache = cache_of(self.make_layer())
        entry_state = self.recurrent(entry_cache)
        before = entry_state.clone()

        adopted = copy_cache(entry_cache)
        adopted.update_recurrent_state(torch.full((1, 2, 4), 7.0), 0)

        self.assertTrue(torch.equal(entry_state, before))
        self.assertFalse(torch.equal(self.recurrent(adopted), before))

    def test_the_conv_state_is_copied_too(self):
        entry_cache = cache_of(self.make_layer())
        entry_conv = self.conv(entry_cache)

        adopted = copy_cache(entry_cache)

        self.assertNotEqual(self.conv(adopted).data_ptr(), entry_conv.data_ptr())

    def test_the_per_state_bookkeeping_is_not_shared(self):
        entry_cache = cache_of(LinearAttentionLayer())
        adopted = copy_cache(entry_cache)

        adopted.update_recurrent_state(torch.ones(1, 2, 4), 0)

        self.assertFalse(entry_cache.layers[0].is_recurrent_states_initialized[0])
        self.assertTrue(adopted.layers[0].is_recurrent_states_initialized[0])


class StoredEntryTests(unittest.TestCase):
    """A store is taken by reference, so the job's later appends must not reach
    back into it either."""

    def test_the_job_growing_past_the_boundary_does_not_change_the_entry(self):
        prompt_cache = PromptCache(lambda: 300, lambda: 100000)
        job_cache = DynamicCache(config=PretrainedConfig(num_hidden_layers=2))
        for idx in range(2):
            job_cache.update(kv(4), kv(4), idx)

        cache_id = b"\x01" * 32
        prompt_cache.store(
            cache_id, job_cache, "node-a", "model-1", ["proc"], 0, 1, MIN_CACHE_TOKENS
        )

        for idx in range(2):
            job_cache.update(kv(5), kv(5), idx)

        entry = prompt_cache.lookup(cache_id, "node-a", "model-1", ["proc"], 0, 1)
        self.assertEqual(entry.cache.get_seq_length(), 4)
        self.assertEqual(job_cache.get_seq_length(), 9)


if __name__ == "__main__":
    unittest.main()
