import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.jobs.job_time import JobTime
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.timing_stats import TimingStats
from language_pipes.util.enums import ComputeStep


def make_pass_times(node_ids=("node-a", "node-b")):
    """Timings for one pass: an embed on the origin, then a layer set per node."""
    times = []
    embed = JobTime(node_id=node_ids[0], is_embed=True)
    embed.set_send_time()
    times.append(embed)
    for node_id in node_ids:
        layer = JobTime(node_id=node_id, start_layer=0, end_layer=4)
        layer.set_send_time()
        times.append(layer)
    return times


def run_origin_pass(origin: TimingStats, is_prefill=False, token_count=1):
    """Finalize one pass on the origin and return what it hands to the pipe."""
    origin.current_times = make_pass_times()
    if is_prefill:
        origin.finalize_prefill_chunk(token_count)
    else:
        origin.finalize_token()
    return origin.completed_pass


class RelayTimingTests(unittest.TestCase):
    """Nodes without the end model never run the head, so they only get their
    stats from the completed passes the origin sends along."""

    def test_relay_records_decode_passes(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        for _ in range(3):
            relay.receive_network_job(make_pass_times(), run_origin_pass(origin))

        self.assertEqual(len(relay.output_times.token_ms), 3)
        self.assertEqual(relay.output_times.token_counts, [1, 1, 1])
        self.assertGreater(relay.output_times.get_tokens_per_second(), 0)
        self.assertGreater(relay.output_times.get_avg_embed_time(), 0)
        self.assertGreater(relay.output_times.get_avg_layer_time(), 0)

    def test_relay_splits_prefill_from_decode(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        relay.receive_network_job([], run_origin_pass(origin, is_prefill=True, token_count=512))
        relay.receive_network_job([], run_origin_pass(origin))

        self.assertEqual(relay.prefill_times.token_counts, [512])
        self.assertEqual(relay.output_times.token_counts, [1])
        self.assertGreater(relay.prefill_times.get_tokens_per_second(), 0)
        self.assertGreater(relay.output_times.get_tokens_per_second(), 0)

    def test_relay_speeds_match_the_origin(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        for _ in range(3):
            relay.receive_network_job(make_pass_times(), run_origin_pass(origin))

        self.assertEqual(
            relay.output_times.get_tokens_per_second(),
            origin.output_times.get_tokens_per_second()
        )

    def test_pass_is_only_recorded_once(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        # A node hosting two layer ranges is handed the same pass twice
        completed = run_origin_pass(origin)
        relay.receive_network_job(make_pass_times(), completed)
        relay.receive_network_job(make_pass_times(), completed)

        self.assertEqual(len(relay.output_times.token_ms), 1)

    def test_relay_forwards_the_pass_it_received(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        completed = run_origin_pass(origin)
        relay.receive_network_job(make_pass_times(), completed)

        self.assertIs(relay.completed_pass, completed)

    def test_origin_ignores_its_own_pass_coming_back(self):
        origin = TimingStats("job-1")

        run_origin_pass(origin)
        # The last node on the pipe hands the job back with that same pass attached
        origin.receive_network_job(make_pass_times(), origin.completed_pass)

        self.assertEqual(len(origin.output_times.token_ms), 1)

    def test_relay_joining_mid_job_records_from_where_it_starts(self):
        origin = TimingStats("job-1")
        relay = TimingStats("job-1")

        for _ in range(3):
            run_origin_pass(origin)
        relay.receive_network_job(make_pass_times(), origin.completed_pass)

        self.assertEqual(len(relay.output_times.token_ms), 1)

    def test_missing_completed_pass_is_a_no_op(self):
        relay = TimingStats("job-1")
        times = make_pass_times()

        relay.receive_network_job(times, None)

        self.assertEqual(relay.current_times, times)
        self.assertEqual(relay.output_times.token_ms, [])
        self.assertIsNone(relay.completed_pass)


class CompletedPassWireTests(unittest.TestCase):
    def test_survives_a_network_round_trip(self):
        origin = TimingStats("job-1")
        completed = run_origin_pass(origin, is_prefill=True, token_count=64)

        network_job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            completed=completed
        )
        restored, _ = NetworkJob.from_bytes(network_job.to_bytes())

        self.assertIsNotNone(restored.completed)
        assert restored.completed is not None
        self.assertEqual(restored.completed.index, completed.index)
        self.assertEqual(restored.completed.token_count, 64)
        self.assertTrue(restored.completed.is_prefill)
        self.assertEqual(len(restored.completed.times), len(completed.times))

        relay = TimingStats("job-1")
        relay.receive_network_job(restored.times, restored.completed)
        self.assertEqual(relay.prefill_times.token_counts, [64])

    def test_absent_pass_round_trips_as_none(self):
        network_job = NetworkJob(
            job_id="job-1",
            pipe_id="pipe-1",
            origin_node_id="node-a",
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[]
        )
        restored, _ = NetworkJob.from_bytes(network_job.to_bytes())

        self.assertIsNone(restored.completed)


if __name__ == "__main__":
    unittest.main()
