import os
import sys
import time
import random
import unittest
sys.path.insert(0, os.path.dirname(__file__))
from base import DSNTestBase, spawn_node, remove_node

class TestDisconnect(DSNTestBase):
    def test_reconnect(self):
        """Node should be removed from peers after disconnect"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        self.assertIn(connector.config.node_id, bootstrap.node.peers())
        connector.stop()
        time.sleep(10)
        self.assertNotIn(connector.config.node_id, bootstrap.node.peers())

    def test_disconnect_and_new_join(self):
        """New nodes should not see disconnected nodes"""
        node1 = spawn_node("node-1", "127.0.0.1")
        node2 = spawn_node("node-2", None, [node1.node.my_con().to_json()])
        node3 = spawn_node("node-3", None, [node1.node.my_con().to_json()])

        time.sleep(1)
        node2.stop()
        remove_node(node2)
        time.sleep(10)
        
        node4 = spawn_node("node-4", None, [node1.node.my_con().to_json()])
        time.sleep(10)
        
        self.assertEqual(["node-1", "node-3", "node-4"], sorted(node1.node.peers()))
        self.assertEqual(["node-1", "node-3", "node-4"], sorted(node3.node.peers()))
        self.assertEqual(["node-1", "node-3", "node-4"], sorted(node4.node.peers()))

    def test_peers_after_disconnect(self):
        """peers() wrapper should reflect disconnected nodes"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        self.assertIn("connector", bootstrap.peers())
        
        connector.stop()
        remove_node(connector)
        
        time.sleep(10)
        
        self.assertNotIn("connector", bootstrap.peers())
        self.assertIn("bootstrap", bootstrap.peers())

    @unittest.skip("Long test")
    def test_churn(self):
        """Network should handle continuous join/leave churn"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        
        stopped = []
        connectors = []
        network_labels = ["bootstrap"]
        for _ in range(5):
            new_connectors = [spawn_node(f"node-{i}", None, [bootstrap.node.my_con().to_json()]) for i in range(len(connectors), len(connectors) + 5)]
            connectors.extend(new_connectors)
            for c in new_connectors:
                network_labels.append(c.config.node_id)
            to_shutdown = random.choice(new_connectors)
            to_shutdown.stop()
            network_labels.remove(to_shutdown.config.node_id)
            stopped.append(to_shutdown)
            time.sleep(6)
            for c in connectors:
                if c.config.node_id not in network_labels:
                    continue
                self.assertEqual(sorted(network_labels), sorted(list(c.node.peers())))

if __name__ == "__main__":
    unittest.main()
