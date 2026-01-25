import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from base import DSNTestBase, spawn_node

class TestStateData(DSNTestBase):
    def test_state_propagation(self):
        """State updates should propagate between nodes"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])

        self.assertEqual(None, bootstrap.node.read_data("connector", "foo"))

        connector.node.update_data("foo", "bar")
        time.sleep(0.5)
        self.assertEqual("bar", bootstrap.node.read_data("connector", "foo"))
        
        bootstrap.node.update_data("bar", "baz")
        time.sleep(0.5)
        self.assertEqual("baz", connector.node.read_data("bootstrap", "bar"))

    def test_multiple_data_updates(self):
        """Multiple data updates should all propagate"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        bootstrap.update_data("key1", "value1")
        bootstrap.update_data("key2", "value2")
        bootstrap.update_data("key3", "value3")
        time.sleep(0.5)
        
        self.assertEqual("value1", connector.read_data("bootstrap", "key1"))
        self.assertEqual("value2", connector.read_data("bootstrap", "key2"))
        self.assertEqual("value3", connector.read_data("bootstrap", "key3"))

    def test_send_to_node_success(self):
        """send_to_node should deliver payload to target node"""
        received_data = []
        
        def recv_fn(data: bytes):
            received_data.append(data)
        
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        bootstrap.set_receive_cb(recv_fn)
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])

        payload = b"Hello, world!"
        resp = connector.node.send_to_node("bootstrap", payload)
        self.assertEqual(resp, "OK")
        
        time.sleep(0.5)
        self.assertEqual(len(received_data), 1)
        self.assertEqual(received_data[0], payload)

    def test_send_to_node_wrapper(self):
        """DSNodeServer.send_to_node wrapper should work"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])

        resp = connector.send_to_node("bootstrap", b"Hello via wrapper")
        self.assertEqual(resp, "OK")


if __name__ == "__main__":
    unittest.main()
