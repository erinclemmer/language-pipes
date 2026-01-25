import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from distributed_state_network.objects.state_packet import StatePacket
from distributed_state_network.objects.hello_packet import HelloPacket

from base import DSNTestBase, spawn_node, remove_node

class TestErrorHandling(DSNTestBase):
    def test_bad_req_data(self):
        """Malformed request data should raise an error"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        try: 
            connector.node.send_http_request(bootstrap.node.my_con(), 1, b'MALFORMED_DATA')
            self.fail("Should throw error for malformed data")
        except Exception as e:
            print(e)

    def test_bad_update_self(self):
        """Node should not handle updates for itself"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        bt_prv_key = bootstrap.node.cred_manager.my_private()
        
        state = StatePacket.create("bootstrap", time.time(), bt_prv_key, {})
        try: 
            bootstrap.node.handle_update(state.to_bytes())
            self.fail("Node should not handle updates for itself")
        except Exception as e:
            print(e)
            self.assertEqual(e.args[0], 406)

    def test_bad_update_unsigned(self):
        """Unsigned state packets should be rejected"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
        state = StatePacket("connector", time.time(), b'', {})
        try:
            bootstrap.node.handle_update(state.to_bytes())
            self.fail("Should not accept unsigned packets")
        except Exception as e:
            print(e)
            self.assertEqual(e.args[0], 401)

    def test_bad_update_stale(self):
        """Stale state updates should be ignored"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        cn_prv_key = connector.node.cred_manager.my_private()

        time_before = time.time() - 10
        state = StatePacket.create("connector", time.time(), cn_prv_key, {"a": "1"})
        bootstrap.node.handle_update(state.to_bytes())

        state = StatePacket.create("connector", time_before, cn_prv_key, {"a": "2"})
        self.assertFalse(bootstrap.node.handle_update(state.to_bytes()))
    
    def test_bad_hello(self):
        """Stale hello packets should not add invalid peers"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector_0 = spawn_node("connector-0", None, [bootstrap.node.my_con().to_json()])
        connector_0.stop()
        remove_node(connector_0)
        connector_1 = spawn_node("connector-1", "127.0.0.1", [bootstrap.node.my_con().to_json()])
        self.assertEqual(sorted(connector_1.node.peers()), ["bootstrap", "connector-1"])

    def test_bad_packets_hello(self):
        """HelloPacket should reject malformed data"""
        try:
            HelloPacket.from_bytes(b'')
            self.fail("Should throw error on bad parse")
        except Exception as e:
            print(e)

        try:
            HelloPacket.from_bytes(b'Random data')
            self.fail("Should throw error on bad parse")
        except Exception as e:
            print(e)

    def test_bad_packets_state(self):
        """StatePacket should reject malformed data"""
        try:
            StatePacket.from_bytes(b'')
            self.fail("Should throw error on bad parse")
        except Exception as e:
            print(e)

        try:
            StatePacket.from_bytes(b'Random data')
            self.fail("Should throw error on bad parse")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    unittest.main()
