import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from transformers import PretrainedConfig

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_cancel import JobCancel
from language_pipes.jobs.job_receiver import CANCEL_PROTOCOL, JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.byte_helper import ByteHelper
from language_pipes.util.enums import ComputeStep


def make_network_job(job_id: str) -> bytes:
    return NetworkJob(
        job_id=job_id,
        pipe_id="pipe-1",
        origin_node_id="node-a",
        current_layer=0,
        data=None,
        data_hash=b"",
        compute_step=ComputeStep.LAYER,
        times=[],
    ).to_bytes()


def make_receiver(max_node_jobs: int = 10) -> JobReceiver:
    # is_shutdown returns True so the background runner loop exits immediately
    # and never touches the (unused) managers.
    return JobReceiver(
        job_factory=None,   # pyright: ignore[reportArgumentType]
        job_tracker=None,   # pyright: ignore[reportArgumentType]
        pipe_manager=None,  # pyright: ignore[reportArgumentType]
        model_manager=None, # pyright: ignore[reportArgumentType]
        is_shutdown=lambda: True,
        get_max_node_jobs=lambda: max_node_jobs,
    )


class ReceiveDataTests(unittest.TestCase):
    def test_queues_job_under_node_id(self):
        receiver = make_receiver()

        receiver.receive_data("node-b", make_network_job("job-1"))

        self.assertIn("node-b", receiver.job_queue)
        self.assertEqual(len(receiver.job_queue["node-b"]), 1)
        self.assertEqual(receiver.job_queue["node-b"][0].job_id, "job-1")

    def test_separate_nodes_get_separate_queues(self):
        receiver = make_receiver()

        receiver.receive_data("node-b", make_network_job("job-1"))
        receiver.receive_data("node-c", make_network_job("job-2"))

        self.assertEqual(len(receiver.job_queue["node-b"]), 1)
        self.assertEqual(len(receiver.job_queue["node-c"]), 1)

    def test_ignores_duplicate_job_ids_from_same_node(self):
        receiver = make_receiver()

        receiver.receive_data("node-b", make_network_job("job-1"))
        receiver.receive_data("node-b", make_network_job("job-1"))

        self.assertEqual(len(receiver.job_queue["node-b"]), 1)

    def test_rejects_jobs_beyond_node_limit(self):
        receiver = make_receiver(max_node_jobs=2)

        # Limit is 2; the guard rejects once the queue already holds more than
        # the limit, so jobs 0..2 are accepted and the next one raises.
        receiver.receive_data("node-b", make_network_job("job-0"))
        receiver.receive_data("node-b", make_network_job("job-1"))
        receiver.receive_data("node-b", make_network_job("job-2"))

        with self.assertRaises(Exception):
            receiver.receive_data("node-b", make_network_job("job-3"))

    def test_limit_is_per_node(self):
        receiver = make_receiver(max_node_jobs=2)

        # Fill node-b to its limit, a different node is unaffected.
        receiver.receive_data("node-b", make_network_job("b-0"))
        receiver.receive_data("node-b", make_network_job("b-1"))
        receiver.receive_data("node-b", make_network_job("b-2"))

        receiver.receive_data("node-c", make_network_job("c-0"))
        self.assertEqual(len(receiver.job_queue["node-c"]), 1)


class FakeRouter:
    def __init__(self, node_id: str):
        self._node_id = node_id
        self.sent = []

    def node_id(self) -> str:
        return self._node_id

    def send_to_node(self, node_id: str, data: bytes):
        self.sent.append((node_id, data))

    def receive_data(self, data: bytes):
        self.sent.append((self._node_id, data))


class FakePipeManager:
    def __init__(self, router: FakeRouter):
        self.router_pipes = type("FakeRouterPipes", (), {"router": router})()


def make_cancel_receiver(node_id: str = "node-a"):
    """Receiver wired to a tracker and a router that records what it sends."""
    router = FakeRouter(node_id)
    tracker = JobTracker()
    tracker.shutdown = True
    receiver = JobReceiver(
        job_factory=None,   # pyright: ignore[reportArgumentType]
        job_tracker=tracker,
        pipe_manager=FakePipeManager(router),  # pyright: ignore[reportArgumentType]
        model_manager=None, # pyright: ignore[reportArgumentType]
        is_shutdown=lambda: True,
        get_max_node_jobs=lambda: 10,
    )
    return receiver, tracker, router


def make_pending_job(
    tracker: JobTracker,
    job_id: str = "job-1",
    origin_node_id: str = "node-a",
    pipe_id: str = "pipe-1",
    model_id: str = "model-1",
    key: str = "network"
) -> Job:
    job = Job(
        origin_node_id=origin_node_id,
        messages=[],
        pipe_id=pipe_id,
        model_id=model_id,
        config=PretrainedConfig(num_hidden_layers=1),
    )
    job.job_id = job_id
    tracker.jobs_pending.setdefault(key, []).append(job)
    return job


def read_cancel(data: bytes) -> JobCancel:
    bts = ByteHelper(data)
    assert bts.read_int() == CANCEL_PROTOCOL
    return JobCancel.from_bytes(bts.read_bytes())


class CancelPipeJobsTests(unittest.TestCase):
    def test_cancels_jobs_running_on_the_pipe(self):
        receiver, tracker, _ = make_cancel_receiver()
        job = make_pending_job(tracker)

        receiver.cancel_pipe_jobs(["pipe-1"], "layers for model-1 unloaded")

        self.assertEqual(job.cancel_reason, "layers for model-1 unloaded")
        self.assertIsNone(tracker.get_job("job-1"))

    def test_leaves_jobs_on_other_pipes_alone(self):
        receiver, tracker, _ = make_cancel_receiver()
        job = make_pending_job(tracker, pipe_id="pipe-2")

        receiver.cancel_pipe_jobs(["pipe-1"], "layers for model-1 unloaded")

        self.assertIsNone(job.cancel_reason)
        self.assertIsNotNone(tracker.get_job("job-1"))

    def test_notifies_origin_node_of_cancel(self):
        receiver, tracker, router = make_cancel_receiver("node-a")
        make_pending_job(tracker, origin_node_id="node-b")

        receiver.cancel_pipe_jobs(["pipe-1"], "layers for model-1 unloaded")

        self.assertEqual(len(router.sent), 1)
        node_id, data = router.sent[0]
        cancel = read_cancel(data)
        self.assertEqual(node_id, "node-b")
        self.assertEqual(cancel.job_id, "job-1")
        self.assertEqual(cancel.pipe_id, "pipe-1")
        self.assertEqual(cancel.reason, "layers for model-1 unloaded")

    def test_does_not_notify_when_the_job_started_here(self):
        receiver, tracker, router = make_cancel_receiver("node-a")
        make_pending_job(tracker, origin_node_id="node-a")

        receiver.cancel_pipe_jobs(["pipe-1"], "layers for model-1 unloaded")

        self.assertEqual(router.sent, [])

    def test_drops_queued_packets_for_the_canceled_job(self):
        receiver, tracker, _ = make_cancel_receiver()
        make_pending_job(tracker)
        receiver.receive_data("node-b", make_network_job("job-1"))
        receiver.receive_data("node-b", make_network_job("job-2"))

        receiver.cancel_pipe_jobs(["pipe-1"], "layers for model-1 unloaded")

        queued = [j.job_id for j in receiver.job_queue.get("node-b", [])]
        self.assertEqual(queued, ["job-2"])


class CancelModelJobsTests(unittest.TestCase):
    def test_cancels_jobs_this_node_started_for_the_model(self):
        receiver, tracker, _ = make_cancel_receiver("node-a")
        job = make_pending_job(tracker, origin_node_id="node-a")

        receiver.cancel_model_jobs("model-1", "end model for model-1 unloaded")

        self.assertEqual(job.cancel_reason, "end model for model-1 unloaded")

    def test_leaves_other_nodes_jobs_alone(self):
        # Another node's job does not run through our end model, so unloading
        # ours says nothing about whether that job can continue.
        receiver, tracker, _ = make_cancel_receiver("node-a")
        job = make_pending_job(tracker, origin_node_id="node-b")

        receiver.cancel_model_jobs("model-1", "end model for model-1 unloaded")

        self.assertIsNone(job.cancel_reason)


class ReceiveCancelTests(unittest.TestCase):
    def test_cancels_the_named_job(self):
        receiver, tracker, _ = make_cancel_receiver("node-a")
        job = make_pending_job(tracker, origin_node_id="node-a")

        receiver.receive_cancel("node-b", JobCancel("job-1", "pipe-1", "layers unloaded").to_bytes())

        self.assertEqual(job.cancel_reason, "layers unloaded")
        self.assertIsNone(tracker.get_job("job-1"))

    def test_ignores_cancel_for_a_different_pipe(self):
        receiver, tracker, _ = make_cancel_receiver("node-a")
        job = make_pending_job(tracker, pipe_id="pipe-1")

        receiver.receive_cancel("node-b", JobCancel("job-1", "pipe-9", "layers unloaded").to_bytes())

        self.assertIsNone(job.cancel_reason)

    def test_ignores_unparseable_payload(self):
        receiver, _, _ = make_cancel_receiver()

        receiver.receive_cancel("node-b", b"not a cancel packet")

    def test_forwards_cancel_toward_the_origin(self):
        receiver, tracker, router = make_cancel_receiver("node-a")
        make_pending_job(tracker, origin_node_id="node-c")

        receiver.receive_cancel("node-b", JobCancel("job-1", "pipe-1", "layers unloaded").to_bytes())

        self.assertEqual([node_id for node_id, _ in router.sent], ["node-c"])


class JobCancelPacketTests(unittest.TestCase):
    def test_round_trips(self):
        cancel = JobCancel("job-1", "pipe-1", "layers for model-1 unloaded")

        parsed = JobCancel.from_bytes(cancel.to_bytes())

        self.assertEqual(parsed.job_id, "job-1")
        self.assertEqual(parsed.pipe_id, "pipe-1")
        self.assertEqual(parsed.reason, "layers for model-1 unloaded")


if __name__ == "__main__":
    unittest.main()
