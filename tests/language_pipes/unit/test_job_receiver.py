import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.network_job import NetworkJob
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


if __name__ == "__main__":
    unittest.main()
