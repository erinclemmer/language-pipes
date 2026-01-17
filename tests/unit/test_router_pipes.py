import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.modeling.llm_meta_data import LlmMetadata
from language_pipes.modeling.meta_model import MetaModel
from language_pipes.pipes.router_pipes import RouterPipes


class FakeConnection:
    def __init__(self, address: str):
        self.address = address


class FakeStateNetworkNode:
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._data = {}
        self._peers = []
        self.logger = None

    def node_id(self):
        return self._node_id

    def read_data(self, node_id: str, key: str):
        return self._data.get((node_id, key))

    def update_data(self, key: str, value: str):
        self._data[(self._node_id, key)] = value

    def peers(self):
        return self._peers

    def connection_from_node(self, node_id: str):
        return FakeConnection("127.0.0.1")

    def add_peer(self, peer_id: str, models=None):
        self._peers.append(peer_id)
        if models is None:
            models = []
        self._data[(peer_id, "models")] = json.dumps([m.to_json() for m in models])


def make_computed():
    metadata = LlmMetadata()
    metadata.embed_size = 128
    metadata.head_size = 256
    metadata.avg_layer_size = 64
    metadata.embed_hash = "embed"
    metadata.head_hash = "head"
    metadata.layer_hashes=["l0", "l1", "l2", "l3"]
    return metadata

class RouterPipesTests(unittest.TestCase):
    def test_add_and_get_pipe_by_pipe_id(self):
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)
        meta_data = make_computed()
        model = MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data)

        router.add_model_to_network(model)
        node.add_peer("node-a", [model])

        pipe = router.get_pipe_by_pipe_id("pipe-1")

        self.assertIsNotNone(pipe)
        self.assertEqual(pipe.pipe_id, "pipe-1")
        self.assertEqual(pipe.model_id, "model-1")

    def test_pipes_for_model_filters_complete(self):
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)
        meta_data = make_computed()
        node.add_peer(
            "node-a",
            [
                MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
                MetaModel("p2", 2, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
                MetaModel("p3", 0, 1, False, "node-c", "pipe-2", "model-1", 4, meta_data),
            ],
        )

        completed = router.pipes_for_model("model-1", True)
        loading = router.pipes_for_model("model-1", False)

        self.assertEqual([p.pipe_id for p in completed], ["pipe-1"])
        self.assertEqual([p.pipe_id for p in loading], ["pipe-2"])

    def test_update_model_replaces_existing(self):
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)
        meta_data = make_computed()
        model = MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data)
        router.add_model_to_network(model)

        updated = MetaModel("p1", 0, 1, False, "node-a", "pipe-1", "model-1", 4, meta_data)
        router.update_model(updated)

        stored = json.loads(node.read_data("node-a", "models"))
        self.assertEqual(stored[0]["loaded"], False)

    def test_get_pipe_by_model_id_returns_complete_pipe(self):
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)
        meta_data = make_computed()
        node.add_peer(
            "node-a",
            [
                MetaModel("p1", 0, 1, True, "node-a", "pipe-1", "model-1", 4, meta_data),
                MetaModel("p2", 2, 3, True, "node-b", "pipe-1", "model-1", 4, meta_data),
            ],
        )

        pipe = router.get_pipe_by_model_id("model-1")

        self.assertIsNotNone(pipe)
        self.assertEqual(pipe.pipe_id, "pipe-1")


if __name__ == "__main__":
    unittest.main()
