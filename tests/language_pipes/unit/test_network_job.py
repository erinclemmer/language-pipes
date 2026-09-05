import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.jobs.network_job import NetworkJob, JobTime
from language_pipes.util.enums import ComputeStep


class NetworkJobTests(unittest.TestCase):
    def test_layer_time_round_trip(self):
        layer_time = JobTime(
            node_id="node-a",
            is_embed=True,
            is_head=False,
            start_layer=0,
            end_layer=3,
        )
        layer_time.set_send_time()

        restored = JobTime.from_bytes(layer_time.to_bytes())

        self.assertEqual(restored.node_id, "node-a")
        self.assertTrue(restored.is_embed)
        self.assertFalse(restored.is_head)
        self.assertEqual(restored.start_layer, 0)
        self.assertEqual(restored.end_layer, 3)

    def test_network_job_round_trip(self):
        layer_time = JobTime(node_id="node-a", start_layer=0, end_layer=1)
        layer_time.set_send_time()
        job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=2,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[layer_time]
        )

        restored, _ = NetworkJob.from_bytes(job.to_bytes())

        self.assertEqual(restored.job_id, "job-1")
        self.assertEqual(restored.pipe_id, "pipe-1")
        self.assertEqual(restored.origin_node_id, "node-a")
        self.assertEqual(restored.current_layer, 2)
        self.assertEqual(restored.compute_step, ComputeStep.LAYER)
        self.assertEqual(len(restored.times), 1)
        self.assertEqual(restored.times[0].node_id, "node-a")

    def test_pass_index_round_trips(self):
        job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=7
        )

        restored, _ = NetworkJob.from_bytes(job.to_bytes())

        self.assertEqual(restored.pass_idx, 7)

    def test_payload_from_a_peer_without_pass_numbers_reads_zero(self):
        job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=7
        )

        # `pass_idx` is the last field written, as a 4-byte int. Cutting it off
        # gives the bytes an older peer would have produced.
        old_payload = job.to_bytes()[:-4]
        restored, valid = NetworkJob.from_bytes(old_payload)

        self.assertTrue(valid)
        self.assertEqual(restored.pass_idx, 0)
        self.assertEqual(restored.job_id, "job-1")


if __name__ == "__main__":
    unittest.main()
