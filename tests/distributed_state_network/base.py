import os
import sys
import shutil
import unittest
from typing import List, Dict, Optional, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from distributed_state_network import DSNodeServer, DSNodeConfig

current_port = 8000
nodes = []

aes_key = DSNodeServer.generate_key()


def spawn_node(
    node_id: str,
    network_ip: Optional[str],
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


def get_nodes():
    global nodes
    return nodes


def remove_node(node):
    global nodes
    nodes.remove(node)


class DSNTestBase(unittest.TestCase):
    def tearDown(self):
        global nodes
        for n in nodes:
            n.stop()
        nodes = []

        if os.path.exists('certs'):
            shutil.rmtree('certs')

        if os.path.exists('credentials'):
            shutil.rmtree('credentials')
