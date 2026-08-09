import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from transformers import PretrainedConfig

from language_pipes.jobs.job import Job
from language_pipes.util.enums import ComputeStep, JobStatus
from language_pipes.util.utils import CHUNK_SIZE


def make_job():
    return Job(
        origin_node_id="node-a",
        messages=[],
        pipe_id="pipe-1",
        model_id="model-1",
        config=PretrainedConfig(num_hidden_layers=1),
    )


def make_relay(origin: Job) -> Job:
    """The same job as tracked by a node that hosts layers but no end model."""
    relay = make_job()
    relay.job_id = origin.job_id
    relay.pipe_id = origin.pipe_id
    relay.origin_node_id = origin.origin_node_id
    return relay


class JobOutputTests(unittest.TestCase):
    def test_set_output_completes_when_token_matches_int_eos(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD

        job.set_output(token=42, eos_token=42)

        self.assertEqual(job.status, JobStatus.COMPLETED)

    def test_set_output_completes_when_token_in_eos_collection(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD

        job.set_output(token=128001, eos_token={2, 128001})

        self.assertEqual(job.status, JobStatus.COMPLETED)

    def test_set_output_continues_when_eos_is_none(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD

        job.set_output(token=7, eos_token=None)

        self.assertEqual(job.status, JobStatus.IN_PROGRESS)


class JobSendUpdateTests(unittest.TestCase):
    def test_send_update_returns_false_without_calling_update_when_stale(self):
        job = make_job()
        job.stale = True
        job.update = lambda j: self.fail("update() should not be called once stale")

        self.assertFalse(job.send_update())

    def test_send_update_calls_update_when_not_stale(self):
        job = make_job()
        calls = []
        job.update = lambda j: calls.append(j) or True

        self.assertTrue(job.send_update())
        self.assertEqual(calls, [job])


class JobCompletedPassTests(unittest.TestCase):
    """The finished pass rides along on the next network job so downstream nodes
    can report speeds without ever running the head themselves."""

    def test_outgoing_job_carries_the_last_finished_pass(self):
        job = make_job()
        job.timing_stats.add_embed_time("node-a")
        job.timing_stats.set_send_time()
        job.timing_stats.finalize_token()

        network_job = job.to_network_job()

        self.assertIsNotNone(network_job.completed)
        assert network_job.completed is not None
        self.assertEqual(network_job.completed.token_count, 1)
        self.assertFalse(network_job.completed.is_prefill)

    def test_incoming_job_records_the_pass_on_this_node(self):
        origin = make_job()
        origin.timing_stats.add_embed_time("node-a")
        origin.timing_stats.set_send_time()
        origin.timing_stats.finalize_token()

        relay = make_relay(origin)

        self.assertTrue(relay.receive_network_job(origin.to_network_job(), "node-b"))
        self.assertEqual(len(relay.timing_stats.output_times.token_ms), 1)


class JobProgressTests(unittest.TestCase):
    """A node hosting only layers never runs the tokenizer or the head, so its own
    `current_token` and `chunking` stay at zero for the life of the job."""

    def test_relay_reports_the_decode_token_the_origin_reached(self):
        origin = make_job()
        origin.current_token = 7
        origin.prompt_tokens = 40
        relay = make_relay(origin)

        relay.receive_network_job(origin.to_network_job(), "node-b")

        self.assertEqual(relay.display_progress().current_token, 7)
        self.assertEqual(relay.display_progress().prompt_tokens, 40)
        self.assertFalse(relay.display_progress().prefilling)

    def test_relay_reports_prefill_position(self):
        origin = make_job()
        origin.prompt_tokens = CHUNK_SIZE * 4
        origin.init_chunking()
        origin.chunking.advance()
        relay = make_relay(origin)

        relay.receive_network_job(origin.to_network_job(), "node-b")

        progress = relay.display_progress()
        self.assertTrue(progress.prefilling)
        self.assertEqual(progress.prefill_tokens, CHUNK_SIZE)
        self.assertEqual(progress.prompt_tokens, CHUNK_SIZE * 4)

    def test_relay_routing_state_is_left_untouched(self):
        # `set_layer` asks `chunking.has_more()` whether a finished layer pass goes
        # back to the origin, so a mirrored chunk state would misroute it
        origin = make_job()
        origin.prompt_tokens = CHUNK_SIZE * 4
        origin.init_chunking()
        relay = make_relay(origin)

        relay.receive_network_job(origin.to_network_job(), "node-b")

        self.assertEqual(relay.current_token, 0)
        self.assertFalse(relay.chunking.is_active())
        self.assertFalse(relay.chunking.has_more())

    def test_origin_reports_its_own_live_state(self):
        origin = make_job()
        origin.current_token = 3
        # The last node on the pipe hands the job back carrying the origin's own
        # report from when it was sent, one token behind where the origin now is
        sent = origin.to_network_job()
        origin.current_token = 4

        origin.receive_network_job(sent, origin.origin_node_id)

        self.assertEqual(origin.display_progress().current_token, 4)

    def test_older_peer_leaves_the_last_reading_in_place(self):
        origin = make_job()
        origin.current_token = 2
        relay = make_relay(origin)
        relay.receive_network_job(origin.to_network_job(), "node-b")

        stale = origin.to_network_job()
        stale.progress = None
        relay.receive_network_job(stale, "node-b")

        self.assertEqual(relay.display_progress().current_token, 2)


class JobPastSeenTokensTests(unittest.TestCase):
    """`past_seen_tokens` has to be derived from job state, not from `job.cache`:
    a node only populates the cache layers it hosts, and a hybrid
    linear-attention stack can leave it with no layer that tracks sequence
    length at all (Qwen3.5 opens with three `linear_attention` layers)."""

    def test_zero_before_the_first_chunk(self):
        job = make_job()
        job.prompt_tokens = 10
        job.init_chunking()

        self.assertEqual(job.past_seen_tokens(), 0)

    def test_tracks_finished_chunks_during_chunked_prefill(self):
        job = make_job()
        job.prompt_tokens = CHUNK_SIZE * 2 + 5
        job.init_chunking()

        self.assertEqual(job.past_seen_tokens(), 0)
        job.chunking.advance()
        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE)
        job.chunking.advance()
        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 2)

    def test_counts_every_token_but_the_next_during_decode(self):
        job = make_job()
        job.prompt_tokens = 10
        job.input_ids = list(range(10))
        job.init_chunking()

        # First generated token appended, decode begins.
        job.current_token = 1
        job.input_ids.append(99)
        self.assertEqual(job.past_seen_tokens(), 10)

        job.current_token = 2
        job.input_ids.append(98)
        self.assertEqual(job.past_seen_tokens(), 11)

    def test_decode_is_unaffected_by_stale_chunk_state(self):
        """`_state_embed` keeps calling `chunking.advance()` while chunking is
        active, so chunk state is meaningless once decoding starts."""
        job = make_job()
        job.prompt_tokens = CHUNK_SIZE * 2
        job.input_ids = list(range(CHUNK_SIZE * 2))
        job.init_chunking()
        job.chunking.advance()
        job.chunking.advance()
        job.chunking.advance()

        job.current_token = 1
        job.input_ids.append(99)

        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 2)


if __name__ == "__main__":
    unittest.main()
