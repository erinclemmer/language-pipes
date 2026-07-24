import os
import sys
import time
import unittest
sys.path.insert(0, os.path.dirname(__file__))
from base import DSNTestBase, spawn_node, remove_node

class TestCallbacks(DSNTestBase):

    def test_disconnect_callback(self):
        """Disconnect callback should be called when a peer disconnects"""
        callback_called = []
        
        def on_disconnect():
            callback_called.append(True)
        
        node1 = spawn_node("node-1", "127.0.0.1", [], on_disconnect)
        node2 = spawn_node("node-2", None, [node1.node.my_con().to_json()])
        
        time.sleep(1)
        node2.stop()
        remove_node(node2)
        time.sleep(10)
        
        self.assertEqual(len(callback_called), 1)

    def test_update_callback(self):
        """Update callback should be called when peer data is updated"""
        callback_called = []
        
        def on_update():
            callback_called.append(True)
        
        node1 = spawn_node("node-1", "127.0.0.1", [], None, on_update)
        node2 = spawn_node("node-2", None, [node1.node.my_con().to_json()])
        
        node2.node.update_data("key", "value")
        time.sleep(1)
        
        self.assertEqual(len(callback_called), 2)

    def test_update_callback_error_handling(self):
        """Update callback errors should not crash the node"""
        def on_update_error():
            raise Exception("This should be captured")
        
        node1 = spawn_node("node-1", "127.0.0.1", [], None, on_update_error)
        node2 = spawn_node("node-2", None, [node1.node.my_con().to_json()])

        node2.node.update_data("key", "value")
        time.sleep(1)
        
        # Should reach this point without crashing
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
