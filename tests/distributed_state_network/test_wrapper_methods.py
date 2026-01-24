import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from base import DSNTestBase, spawn_node, remove_node

class TestWrapperMethods(DSNTestBase):
    def test_peers_wrapper(self):
        """DSNodeServer.peers() should return peer list"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        peers = bootstrap.peers()
        self.assertIsInstance(peers, list)
        self.assertIn("bootstrap", peers)
        
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        bootstrap_peers = bootstrap.peers()
        connector_peers = connector.peers()
        
        self.assertIn("bootstrap", bootstrap_peers)
        self.assertIn("connector", bootstrap_peers)
        self.assertIn("bootstrap", connector_peers)
        self.assertIn("connector", connector_peers)

    def test_read_data_wrapper(self):
        """DSNodeServer.read_data() should read peer data"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        result = bootstrap.read_data("connector", "nonexistent_key")
        self.assertIsNone(result)
        
        connector.node.update_data("test_key", "test_value")
        time.sleep(0.5)
        
        result = bootstrap.read_data("connector", "test_key")
        self.assertEqual("test_value", result)
        
        bootstrap.node.update_data("own_key", "own_value")
        result = bootstrap.read_data("bootstrap", "own_key")
        self.assertEqual("own_value", result)

    def test_update_data_wrapper(self):
        """DSNodeServer.update_data() should update and propagate data"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        bootstrap.update_data("wrapper_key", "wrapper_value")
        time.sleep(0.5)
        
        result = connector.read_data("bootstrap", "wrapper_key")
        self.assertEqual("wrapper_value", result)
        
        result = bootstrap.read_data("bootstrap", "wrapper_key")
        self.assertEqual("wrapper_value", result)
        
        bootstrap.update_data("wrapper_key", "updated_value")
        time.sleep(0.5)
        
        result = connector.read_data("bootstrap", "wrapper_key")
        self.assertEqual("updated_value", result)

    def test_is_shut_down_wrapper(self):
        """DSNodeServer.is_shut_down() should reflect shutdown state"""
        node = spawn_node("shutdown_test", "127.0.0.1")
        
        self.assertFalse(node.is_shut_down())
        
        node.stop()
        self.assertTrue(node.is_shut_down())
        
        remove_node(node)

    def test_node_id_wrapper(self):
        """DSNodeServer.node_id() should return configured node ID"""
        node = spawn_node("test_node_id", "127.0.0.1")
        self.assertEqual("test_node_id", node.node_id())
        
        node2 = spawn_node("another_node", "127.0.0.1")
        self.assertEqual("another_node", node2.node_id())
        
        node3 = spawn_node("node-with-dashes", "127.0.0.1")
        self.assertEqual("node-with-dashes", node3.node_id())

    def test_wrapper_methods_consistency(self):
        """Wrapper methods should be consistent with direct node access"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        self.assertEqual(bootstrap.peers(), bootstrap.node.peers())
        self.assertEqual(connector.peers(), connector.node.peers())
        
        self.assertEqual(bootstrap.node_id(), bootstrap.config.node_id)
        self.assertEqual(connector.node_id(), connector.config.node_id)
        
        self.assertEqual(bootstrap.is_shut_down(), bootstrap.node.shutting_down)
        
        connector.update_data("test", "value")
        time.sleep(0.5)
        
        self.assertEqual(
            bootstrap.read_data("connector", "test"),
            bootstrap.node.read_data("connector", "test")
        )


if __name__ == "__main__":
    unittest.main()
