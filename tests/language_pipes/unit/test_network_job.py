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

    def test_cache_tags_round_trip(self):
        job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=7,
            attempt=2,
            cache_use_id=b"\x01" * 32,
            cache_use_tokens=384,
            cache_write_id=b"\x02" * 32,
            cache_write_tokens=512,
            cache_reserve_tokens=1600
        )

        restored, _ = NetworkJob.from_bytes(job.to_bytes())

        self.assertEqual(restored.attempt, 2)
        self.assertEqual(restored.cache_use_id, b"\x01" * 32)
        self.assertEqual(restored.cache_use_tokens, 384)
        self.assertEqual(restored.cache_write_id, b"\x02" * 32)
        self.assertEqual(restored.cache_write_tokens, 512)
        self.assertEqual(restored.cache_reserve_tokens, 1600)

    def test_payload_from_a_peer_without_the_appended_fields_reads_empty(self):
        job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=7,
            attempt=2,
            cache_use_id=b"\x01" * 32,
            cache_use_tokens=384,
            cache_write_id=b"\x02" * 32,
            cache_write_tokens=512,
            cache_reserve_tokens=1600
        )

        # Everything after `progress` is appended: four 4-byte ints and two
        # length-prefixed 32-byte IDs. Cutting them all off gives the bytes a
        # peer from before the restart fix and the cache protocol would send.
        appended = 4 + 4 + (4 + 32) + 4 + (4 + 32) + 4 + 4
        restored, valid = NetworkJob.from_bytes(job.to_bytes()[:-appended])

        self.assertTrue(valid)
        self.assertEqual(restored.job_id, "job-1")
        self.assertEqual(restored.pass_idx, 0)
        self.assertEqual(restored.attempt, 0)
        self.assertEqual(restored.cache_use_id, b"")
        self.assertEqual(restored.cache_use_tokens, 0)
        self.assertEqual(restored.cache_write_id, b"")
        self.assertEqual(restored.cache_write_tokens, 0)
        self.assertEqual(restored.cache_reserve_tokens, 0)


if __name__ == "__main__":
    unittest.main()
