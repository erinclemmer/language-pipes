import os
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
from base import DSNTestBase, spawn_node

from distributed_state_network.objects.endpoint import Endpoint

class TestConnectivity(DSNTestBase):
    def test_single_node(self):
        """Single node should see itself in peers"""
        node = spawn_node("one", "127.0.0.1")
        self.assertIn("one", list(node.node.peers()))

    def test_two_nodes(self):
        """Two nodes should discover each other"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        self.assertIn("connector", list(bootstrap.node.peers()))
        self.assertIn("bootstrap", list(bootstrap.node.peers()))
        self.assertIn("connector", list(connector.node.peers()))
        self.assertIn("bootstrap", list(connector.node.peers()))

    def test_many_nodes(self):
        """Many nodes connecting to one bootstrap should all discover each other"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connectors = [spawn_node(f"node-{i}", None, [bootstrap.node.my_con().to_json()]) for i in range(0, 10)]

        boot_peers = list(bootstrap.node.peers())

        for c in connectors:
            peers = c.node.peers()
            self.assertIn(c.config.node_id, boot_peers)
            self.assertIn("bootstrap", list(peers))
            for i in range(0, 10):
                self.assertIn(f"node-{i}", list(peers))

    def test_multi_bootstrap(self):
        """Multiple bootstrap nodes should propagate peer info across the network"""
        bootstraps = [spawn_node(f"bootstrap-{i}", "127.0.0.1") for i in range(0, 3)]
        for i in range(1, len(bootstraps)):
            bootstraps[i].node.bootstrap(bootstraps[i-1].node.my_con())
        
        connectors = []
        for bs in bootstraps:
            new_connectors = [spawn_node(f"node-{i}", None, [bs.node.my_con().to_json()]) for i in range(len(connectors), len(connectors) + 3)]
            connectors.extend(new_connectors)
        
        for ci in connectors:
            peers = ci.node.peers()
            for cj in connectors:
                self.assertIn(cj.config.node_id, peers)
            for b in bootstraps:
                self.assertIn(b.config.node_id, peers)
        
        for bi in bootstraps:
            peers = bi.node.peers()
            for bj in bootstraps:
                self.assertIn(bj.config.node_id, peers)
            for c in connectors:
                self.assertIn(c.config.node_id, peers)

    def test_unreachable_peer_does_not_block_discovery(self):
        """An unreachable entry in the bootstrap node's address book must not hide the peers listed after it"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")

        # A peer the bootstrap node still lists but can no longer be reached.
        # It is deliberately inserted before "reachable" so discovery hits it first.
        bootstrap.node.address_book["dead"] = Endpoint("127.0.0.1", 1)
        reachable = spawn_node("reachable", "127.0.0.1", [bootstrap.node.my_con().to_json()])

        self.assertEqual(
            ["bootstrap", "dead", "reachable"],
            list(bootstrap.node.address_book.keys())
        )

        connector = spawn_node("connector", "127.0.0.1", [bootstrap.node.my_con().to_json()])

        self.assertIn("bootstrap", list(connector.node.peers()))
        self.assertIn("reachable", list(connector.node.peers()))
        self.assertIn("connector", list(reachable.node.peers()))
        # The peer we never reached must not be left behind in the address book
        self.assertNotIn("dead", connector.node.address_book)

    def test_rediscovery_with_already_known_state(self):
        """Discovery must survive peers that reject our update as a duplicate"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        other = spawn_node("other", "127.0.0.1", [bootstrap.node.my_con().to_json()])
        connector = spawn_node("connector", "127.0.0.1", [bootstrap.node.my_con().to_json()])

        connector.update_data("role", "worker")
        self.assertEqual({"role": "worker"}, other.node.node_states["connector"].state_data)

        # "other" now holds an identical copy of our state and answers the UPDATE
        # with an empty body, which must not abort the peer walk.
        connector.node.request_peers("bootstrap")

        self.assertIn("other", list(connector.node.peers()))

    def test_missed_peer_is_rediscovered(self):
        """A node that missed a peer during bootstrap heals on a later tick"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        other = spawn_node("other", "127.0.0.1", [bootstrap.node.my_con().to_json()])
        connector = spawn_node("connector", "127.0.0.1", [bootstrap.node.my_con().to_json()])

        # Simulate a bootstrap that never learned about "other"
        connector.node.node_states.pop("other", None)
        connector.node.address_book.pop("other", None)

        deadline = time.time() + 15
        while time.time() < deadline and "other" not in connector.node.address_book:
            time.sleep(0.25)

        self.assertIn("other", connector.node.address_book)
        self.assertIn("other", list(connector.node.peers()))

    def test_connection_from_node(self):
        """Should be able to look up connection info by node ID"""
        n0 = spawn_node("node-0", "127.0.0.1")
        n1 = spawn_node("node-1", None, [n0.node.my_con().to_json()])
        con = n0.node.connection_from_node("node-1")
        self.assertEqual(con.port, n1.config.port)
        try:
            n0.node.connection_from_node("test")
            self.fail("Should throw error if it can't find a matching node")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    import unittest
    unittest.main()
