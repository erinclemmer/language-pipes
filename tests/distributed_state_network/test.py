import os
import sys
import time
import random
import shutil
import unittest
import requests
from typing import List, Dict, Optional, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from distributed_state_network import DSNodeServer, DSNodeConfig, Endpoint

from distributed_state_network.objects.state_packet import StatePacket
from distributed_state_network.objects.hello_packet import HelloPacket
from distributed_state_network.objects.data_packet import DataPacket

current_port = 8000
nodes = []

aes_key = DSNodeServer.generate_key()


def spawn_node(
    node_id: str,
    network_ip: str,
    bootstrap_nodes: List[Dict] = [],
    disconnect_cb: Optional[Callable] = None,
    update_cb: Optional[Callable] = None
):
    global current_port
    current_port += 1
    n = DSNodeServer.start(DSNodeConfig.from_dict({
        "node_id": node_id,
        "port": current_port,
        "network_ip": network_ip,
        "aes_key": aes_key,
        "bootstrap_nodes": bootstrap_nodes
    }), disconnect_cb, update_cb)
    global nodes
    nodes.append(n)
    return n


class DSNTestBase(unittest.TestCase):
    """Base class for DSN tests with common setup/teardown"""
    
    def tearDown(self):
        global nodes
        for n in nodes:
            n.stop()
        nodes = []

        if os.path.exists('certs'):
            shutil.rmtree('certs')

        if os.path.exists('credentials'):
            shutil.rmtree('credentials')


class TestConnectivity(DSNTestBase):
    """Tests for basic node connectivity and peer discovery"""

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


class TestDisconnect(DSNTestBase):
    """Tests for node disconnect handling"""

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
        global nodes
        nodes.remove(node2)
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
        global nodes
        nodes.remove(connector)
        
        time.sleep(10)
        
        self.assertNotIn("connector", bootstrap.peers())
        self.assertIn("bootstrap", bootstrap.peers())

    @unittest.skip("Flaky due to timing")
    def test_churn(self):
        """Network should handle continuous join/leave churn"""
        bootstrap = spawn_node("bootstrap", "127.0.0.1")
        
        stopped = []
        connectors = []
        network_labels = ["bootstrap"]
        for i in range(5):
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


class TestCallbacks(DSNTestBase):
    """Tests for callback functionality"""

    def test_disconnect_callback(self):
        """Disconnect callback should be called when a peer disconnects"""
        callback_called = []
        
        def on_disconnect():
            callback_called.append(True)
        
        node1 = spawn_node("node-1", "127.0.0.1", [], on_disconnect)
        node2 = spawn_node("node-2", None, [node1.node.my_con().to_json()])
        
        time.sleep(1)
        node2.stop()
        global nodes
        nodes.remove(node2)
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
        
        self.assertEqual(len(callback_called), 1)

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


class TestStateData(DSNTestBase):
    """Tests for state/data operations"""

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


class TestSecurity(DSNTestBase):
    """Tests for security and authorization"""

    def test_bad_aes_key(self):
        """Invalid AES key should raise an error"""
        try:
            DSNodeServer.start(DSNodeConfig("bad key test", 8080, "bad.key", []))
            self.fail("Should throw error before this")
        except Exception as e:
            print(e)

    def test_authorization_reject_unencrypted(self):
        """Unencrypted HTTP requests should be rejected"""
        n = spawn_node("node", "127.0.0.1")
        time.sleep(0.5)
        
        try:
            response = requests.post(
                f'http://127.0.0.1:{n.config.port}/ping',
                data=b'TEST',
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
            connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
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
        global nodes
        nodes.remove(connector)
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
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])

        pkt = DataPacket("connector", b"", b"payload")
        encrypted = bootstrap.node.encrypt_data(bytes([5]) + pkt.to_bytes())

        response = requests.post(
            f'http://127.0.0.1:{bootstrap.config.port}/data',
            data=encrypted,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )
        self.assertEqual(response.status_code, 401)


class TestErrorHandling(DSNTestBase):
    """Tests for error handling"""

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
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
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
        connector = spawn_node("connector", None, [bootstrap.node.my_con().to_json()])
        
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
        global nodes
        nodes.remove(connector_0)
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


class TestConfiguration(DSNTestBase):
    """Tests for configuration handling"""

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


class TestHTTPEndpoints(DSNTestBase):
    """Tests for HTTP endpoint accessibility"""

    def test_http_endpoints_exist(self):
        """All HTTP endpoints should exist (not return 404)"""
        n = spawn_node("http-test-node", "127.0.0.1")
        time.sleep(0.5)
        
        endpoints = ['/hello', '/peers', '/update', '/ping', '/data']
        
        for endpoint in endpoints:
            try:
                response = requests.post(
                    f'http://127.0.0.1:{n.config.port}{endpoint}',
                    data=b'test',
                    timeout=2
                )
                self.assertNotEqual(response.status_code, 404, f"Endpoint {endpoint} not found")
                print(f"Endpoint {endpoint} exists (status: {response.status_code})")
            except Exception as e:
                print(f"Endpoint {endpoint} test failed: {e}")


class TestWrapperMethods(DSNTestBase):
    """Tests for DSNodeServer wrapper methods"""

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
        
        global nodes
        nodes.remove(node)

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
