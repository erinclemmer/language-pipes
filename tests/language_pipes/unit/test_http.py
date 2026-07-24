import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.util.http import _connection_alive


class FakeHandler:
    def __init__(self, connection):
        self.connection = connection


class ConnectionAliveTests(unittest.TestCase):
    def test_true_while_peer_open(self):
        local, peer = socket.socketpair()
        try:
            self.assertTrue(_connection_alive(FakeHandler(local)))
        finally:
            local.close()
            peer.close()

    def test_false_after_peer_closes(self):
        local, peer = socket.socketpair()
        peer.close()
        try:
            self.assertFalse(_connection_alive(FakeHandler(local)))
        finally:
            local.close()

    def test_true_when_handler_has_no_connection(self):
        self.assertTrue(_connection_alive(FakeHandler(None)))
