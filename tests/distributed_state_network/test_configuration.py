import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from distributed_state_network import DSNodeServer, DSNodeConfig

from base import DSNTestBase


class TestConfiguration(DSNTestBase):
    def test_config_dict(self):
        """DSNodeConfig.from_dict should parse config correctly"""
        config_dict = {
            "node_id": "node",
            "port": 8000,
            "aes_key": "XXX",
            "bootstrap_nodes": [
                {
                    "address": "127.0.0.1",
                    "port": 8001
                }
            ]
        }

        config = DSNodeConfig.from_dict(config_dict)
        self.assertEqual(config_dict["node_id"], config.node_id)
        self.assertEqual(config_dict["port"], config.port)
        self.assertEqual(config_dict["aes_key"], config.aes_key)
        self.assertTrue(len(config.bootstrap_nodes) > 0)
        self.assertEqual(config_dict["bootstrap_nodes"][0]["address"], config.bootstrap_nodes[0].address)
        self.assertEqual(config_dict["bootstrap_nodes"][0]["port"], config.bootstrap_nodes[0].port)

    def test_aes_key_generation(self):
        """Generated AES key should be 64 characters"""
        key = DSNodeServer.generate_key()
        self.assertEqual(64, len(key))


if __name__ == "__main__":
    unittest.main()
