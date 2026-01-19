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
            times=[layer_time],
        )

        restored, _ = NetworkJob.from_bytes(job.to_bytes())

        self.assertEqual(restored.job_id, "job-1")
        self.assertEqual(restored.pipe_id, "pipe-1")
        self.assertEqual(restored.origin_node_id, "node-a")
        self.assertEqual(restored.current_layer, 2)
        self.assertEqual(restored.compute_step, ComputeStep.LAYER)
        self.assertEqual(len(restored.times), 1)
        self.assertEqual(restored.times[0].node_id, "node-a")


if __name__ == "__main__":
    unittest.main()
