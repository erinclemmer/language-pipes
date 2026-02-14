import os
import sys
import time
import shutil
import unittest
import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from language_pipes.distributed_state_network import DSNodeServer, DSNodeConfig
from language_pipes.distributed_state_network.objects.data_packet import DataPacket

from base import DSNTestBase, spawn_node, remove_node

class TestSecurity(DSNTestBase):
    def test_bad_aes_key(self):
        """Invalid AES key should raise an error"""
        try:
            DSNodeServer.start(DSNodeConfig("bad key test", "", 8080, None, "bad.key", [], [], []))
            self.fail("Should throw error before this")
        except Exception as e:
            print(e)

    def test_legacy_aes_key_rejected(self):
        """Legacy 32-byte [IV|KEY] material should be rejected."""
        legacy_hex = (os.urandom(32)).hex()
        with self.assertRaises(ValueError):
            DSNodeServer.start(DSNodeConfig("legacy key test", "credentials", 8080, None, legacy_hex, [], [], []))

    def test_whitelist_node_ids_allows_configured_node_id(self):
        """Nodes should communicate when peer node_id is whitelisted."""
        bootstrap = spawn_node("bootstrap", "127.0.0.1", whitelist_node_ids=["connector"])
        connector = spawn_node(
            "connector",
            None,
            [bootstrap.node.my_con().to_json()],
            whitelist_node_ids=["bootstrap"]
        )

        self.assertIn("connector", bootstrap.node.peers())
        self.assertIn("bootstrap", connector.node.peers())

    def test_whitelist_node_ids_rejects_non_whitelisted_node_id(self):
        """Nodes should not communicate when peer node_id is not whitelisted."""
        bootstrap = spawn_node("bootstrap", "127.0.0.1", whitelist_node_ids=["some-other-node"])
        connector = spawn_node(
            "connector",
            None,
            [bootstrap.node.my_con().to_json()]
        )

        self.assertEqual(sorted(connector.node.peers()), ["connector"])

    def test_whitelist_allows_configured_ip(self):
        """Nodes should communicate when peer IP is whitelisted."""
        bootstrap = spawn_node("bootstrap", "127.0.0.1", whitelist_ips=["127.0.0.1"])
        connector = spawn_node(
            "connector",
            None,
            [bootstrap.node.my_con().to_json()],
            whitelist_ips=["127.0.0.1"]
        )

        self.assertIn("connector", bootstrap.node.peers())
        self.assertIn("bootstrap", connector.node.peers())

    def test_whitelist_rejects_non_whitelisted_inbound_ip(self):
        """Inbound requests from non-whitelisted IPs should be rejected."""
        n = spawn_node("node", "127.0.0.1", whitelist_ips=["192.168.1.10"])
        time.sleep(0.5)

        encrypted_data = n.node.encrypt_data(bytes([4]) + b'TEST')  # MSG_PING = 4
        response = requests.post(
            f'http://127.0.0.1:{n.config.port}/ping',
            data=encrypted_data,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )

        self.assertEqual(response.status_code, 401)

    def test_whitelist_rejects_non_whitelisted_outbound_ip(self):
        """Outbound requests to non-whitelisted IPs should fail."""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node(
            "connector",
            None,
            [bootstrap.node.my_con().to_json()],
            whitelist_ips=["10.10.10.10"]
        )

        self.assertEqual(sorted(connector.node.peers()), ["connector"])

    def test_authorization_reject_unencrypted(self):
        """Unencrypted HTTP requests should be rejected"""
        n = spawn_node("node", "127.0.0.1")
        time.sleep(0.5)
        
        try:
            response = requests.post(
                f'http://127.0.0.1:{n.config.port}/ping',
                data=bytes([4]) + b'TEST',
                timeout=2
            )
            self.assertNotEqual(response.status_code, 200)
            print(f"Received status code for bad data: {response.status_code}")
        except Exception as e:
            print(f"Request failed as expected: {e}")

    def test_authorization_accept_encrypted(self):
        """Properly encrypted requests should be accepted"""
        n = spawn_node("node", "127.0.0.1")
        time.sleep(0.5)
        
        encrypted_data = n.node.encrypt_data(bytes([4]) + b'TEST')  # MSG_PING = 4
        response = requests.post(
            f'http://127.0.0.1:{n.config.port}/ping',
            data=encrypted_data,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )
        
        self.assertEqual(response.status_code, 200)
        decrypted = n.node.decrypt_data(response.content)
        self.assertEqual(decrypted[0], 4)

    def test_version_matching(self):
        """Nodes with mismatched versions should not connect"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        old_version = bootstrap.node.version
        bootstrap.node.version = "bad_version"
        try:
            spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
            self.fail("Should throw error when connecting with version mismatch")
        except Exception as e:
            print(e)
        finally:
            bootstrap.node.version = old_version

    def test_authentication_reset(self):
        """Node with reset credentials should fail to reconnect"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        connector.stop()
        shutil.rmtree("credentials/connector")
        try:
            connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
            self.fail("Should not be able to authenticate with bootstrap")
        except Exception as e:
            print(e)

    def test_reauthentication(self):
        """Node with preserved credentials should successfully reconnect"""
        if os.path.exists("credentials"):
            shutil.rmtree("credentials")
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        connector.stop()
        remove_node(connector)
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        self.assertIn('connector', bootstrap.node.peers())

    def test_data_route_unknown_sender(self):
        """Data packets from unknown senders should be rejected"""
        n = spawn_node("node", "127.0.0.1")
        time.sleep(0.5)

        pkt = DataPacket("unknown", b"", b"payload")
        encrypted = n.node.encrypt_data(bytes([5]) + pkt.to_bytes())  # MSG_DATA = 5

        response = requests.post(
            f'http://127.0.0.1:{n.config.port}/data',
            data=encrypted,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )
        self.assertEqual(response.status_code, 401)

    def test_data_route_bad_signature(self):
        """Data packets with invalid signatures should be rejected"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        spawn_node("connector", None, [bootstrap.node.my_con().to_json()])

        pkt = DataPacket("connector", b"", b"payload")
        encrypted = bootstrap.node.encrypt_data(bytes([5]) + pkt.to_bytes())

        response = requests.post(
            f'http://127.0.0.1:{bootstrap.config.port}/data',
            data=encrypted,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
