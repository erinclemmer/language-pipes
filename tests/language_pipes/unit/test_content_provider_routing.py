import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.jobs.cache_packets import CacheReason, CacheStatus
from language_pipes.jobs.job_cancel import CANCEL_PROTOCOL, JobCancel
from language_pipes.jobs.job_receiver import CACHE_PROTOCOL
from language_pipes.util.byte_helper import ByteHelper


class FakeJobReceiver:
    def __init__(self):
        self.jobs = []
        self.cancels = []
        self.cache_status = []

    def receive_data(self, node_id, data):
        self.jobs.append((node_id, data))

    def receive_cancel(self, node_id, data):
        self.cancels.append((node_id, data))

    def receive_cache_status(self, node_id, data):
        self.cache_status.append((node_id, data))


def make_provider():
    config_file = Path(tempfile.mkdtemp()) / "config.toml"
    provider = ContentProvider(config_file, lambda alert: None)
    receiver = FakeJobReceiver()
    provider.job_receiver = receiver  # pyright: ignore[reportAttributeAccessIssue]
    return provider, receiver


def framed(protocol: int, payload: bytes) -> bytes:
    bts = ByteHelper()
    bts.write_int(protocol)
    bts.write_bytes(payload)
    return bts.get_bytes()


class ReceiveDataRoutingTests(unittest.TestCase):
    def test_routes_cancel_protocol_to_the_receiver(self):
        provider, receiver = make_provider()
        cancel = JobCancel("job-1", "pipe-1", "layers for model-1 unloaded")

        provider._receive_data("node-b", framed(CANCEL_PROTOCOL, cancel.to_bytes()))

        self.assertEqual(len(receiver.cancels), 1)
        node_id, data = receiver.cancels[0]
        self.assertEqual(node_id, "node-b")
        self.assertEqual(JobCancel.from_bytes(data).job_id, "job-1")
        self.assertEqual(receiver.jobs, [])

    def test_routes_cache_protocol_to_the_receiver(self):
        provider, receiver = make_provider()
        status = CacheStatus("job-1", "pipe-1", 2, CacheReason.MISS)

        provider._receive_data("node-b", framed(CACHE_PROTOCOL, status.to_bytes()))

        self.assertEqual(len(receiver.cache_status), 1)
        node_id, data = receiver.cache_status[0]
        self.assertEqual(node_id, "node-b")
        parsed = CacheStatus.from_bytes(data)
        self.assertEqual(parsed.job_id, "job-1")
        self.assertEqual(parsed.attempt, 2)
        self.assertEqual(parsed.reason, CacheReason.MISS)
        self.assertEqual(receiver.jobs, [])
        self.assertEqual(receiver.cancels, [])

    def test_routes_job_protocol_to_the_receiver(self):
        provider, receiver = make_provider()

        provider._receive_data("node-b", framed(0, b"job-payload"))

        self.assertEqual(receiver.jobs, [("node-b", b"job-payload")])
        self.assertEqual(receiver.cancels, [])


if __name__ == "__main__":
    unittest.main()
