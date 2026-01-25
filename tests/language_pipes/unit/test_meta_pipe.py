import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.modeling.llm_meta_data import LlmMetadata
from language_pipes.modeling.meta_model import MetaModel
from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import aggregate_models

def make_computed():
    metadata = LlmMetadata()
    metadata.embed_size = 128
    metadata.head_size = 256
    metadata.avg_layer_size = 64
    metadata.embed_hash = "embed"
    metadata.head_hash = "head"
    metadata.layer_hashes=["l0", "l1", "l2", "l3"]
    return metadata


class MetaPipeTests(unittest.TestCase):
    def test_is_complete_with_contiguous_loaded_segments(self):
        meta_data = make_computed()
        segments = [
            MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
            MetaModel("p2", 2, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
        ]
        pipe = MetaPipe("pipe-1", "model-1", segments)

        self.assertTrue(pipe.is_complete())

    def test_is_incomplete_with_gap(self):
        meta_data = make_computed()
        segments = [
            MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
            MetaModel("p2", 3, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
        ]
        pipe = MetaPipe("pipe-1", "model-1", segments)

        self.assertFalse(pipe.is_complete())

    def test_next_start_and_end_layer(self):
        meta_data = make_computed()
        segments = [
            MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
        ]
        pipe = MetaPipe("pipe-1", "model-1", segments)

        self.assertEqual(pipe.next_start_layer(), 2)
        self.assertEqual(pipe.next_end_layer(), 3)

    def test_aggregate_models_sorts_segments(self):
        meta_data = make_computed()
        models = [
            MetaModel("p2", 2, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
            MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
        ]
        pipes = aggregate_models(models)

        self.assertEqual(len(pipes), 1)
        self.assertEqual([s.start_layer for s in pipes[0].segments], [0, 2])

    def test_get_filled_slots_marks_loaded_and_loading(self):
        meta_data = make_computed()
        segments = [
            MetaModel("p1", 0, 1, False, "node-a", "pipe-1", "model-1", 4, meta_data),
            MetaModel("p2", 2, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
        ]
        pipe = MetaPipe("pipe-1", "model-1", segments)

        self.assertEqual(pipe.get_filled_slots(), [1, 1, 2, 2])


if __name__ == '__main__':
    unittest.main()
