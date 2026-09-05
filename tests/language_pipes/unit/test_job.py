import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.pass_sequence import MAX_PASS_RETRIES
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


def make_data(value: float = 0.0) -> JobData:
    return JobData(
        state=torch.full((1, 1), value),
        cache_position=torch.tensor([0]),
        position_ids=torch.tensor([[0]]),
        causal_mask={},
        position_embeddings={},
        shared_kv_states={"full_attention": (torch.full((1, 1), value), torch.full((1, 1), value))}
    )


def make_packet(job: Job, pass_idx: int, step=ComputeStep.LAYER, layer: int = 0, data=None) -> NetworkJob:
    """A pass arriving at a node that hosts layers."""
    return NetworkJob(
        job_id=job.job_id,
        pipe_id=job.pipe_id,
        origin_node_id=job.origin_node_id,
        current_layer=layer,
        data=make_data() if data is None else data,
        data_hash=b"",
        compute_step=step,
        times=[],
        pass_idx=pass_idx
    )


def fill_cache(job: Job):
    """Put one position in the cache, the way computing a pass would."""
    keys = torch.zeros((1, 2, 1, 4))
    job.cache.update(keys, keys.clone(), 0)


def compute_and_send(job: Job, output: JobData):
    """What a layer node does between receiving a pass and forwarding it."""
    fill_cache(job)
    job.data = output
    job.compute_step = ComputeStep.HEAD
    job.current_layer = 0
    job.passes.save(job.data, job.compute_step, job.current_layer)
    job.passes.sent()


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

    def test_a_cached_prefix_is_counted_before_the_first_chunk(self):
        job = make_job()
        job.prompt_tokens = CHUNK_SIZE * 8
        job.caching.prefix_len = CHUNK_SIZE * 4
        job.init_chunking()

        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 4)

    def test_a_cached_prefix_is_added_to_this_jobs_own_chunks(self):
        job = make_job()
        job.prompt_tokens = CHUNK_SIZE * 8
        job.caching.prefix_len = CHUNK_SIZE * 4
        job.init_chunking()

        job.chunking.advance()
        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 5)
        job.chunking.advance()
        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 6)

    def test_the_first_decode_step_after_a_cached_prefix_counts_the_whole_prompt(self):
        job = make_job()
        job.prompt_tokens = CHUNK_SIZE * 8
        job.caching.prefix_len = CHUNK_SIZE * 4
        job.input_ids = list(range(CHUNK_SIZE * 8))
        job.init_chunking()

        job.current_token = 1
        job.input_ids.append(99)

        self.assertEqual(job.past_seen_tokens(), CHUNK_SIZE * 8)


class JobWritePointTests(unittest.TestCase):
    """Block boundaries divide evenly into chunks, so a chunk either lands on a
    write point or on none."""

    def test_a_chunk_ending_on_a_write_point_is_recognized(self):
        job = make_job()
        job.caching.write_points = [512]

        self.assertEqual(job.caching.next_write_point(512), 512)

    def test_a_chunk_ending_anywhere_else_is_not(self):
        job = make_job()
        job.caching.write_points = [512]

        self.assertIsNone(job.caching.next_write_point(480))
        self.assertIsNone(job.caching.next_write_point(544))

    def test_a_job_with_no_write_points_never_matches(self):
        self.assertIsNone(make_job().caching.next_write_point(512))


class JobReplayTests(unittest.TestCase):
    """A packet that fails its hash is bounced to the origin, which sends the
    same pass again. A node that already computed that pass must resend what it
    produced: its keys and values are in the cache already, and computing again
    would append them a second time."""

    def receive_and_compute(self, relay: Job, pass_idx: int) -> JobData:
        self.assertTrue(relay.receive_network_job(make_packet(relay, pass_idx), "node-b"))
        self.assertFalse(relay.passes.replaying)
        output = make_data(float(pass_idx))
        compute_and_send(relay, output)
        return output

    def test_repeat_of_the_last_pass_resends_the_saved_output(self):
        relay = make_relay(make_job())
        output = self.receive_and_compute(relay, 1)

        self.assertTrue(relay.receive_network_job(make_packet(relay, 1), "node-b"))

        self.assertTrue(relay.passes.replaying)
        self.assertIs(relay.data, output)
        self.assertEqual(relay.compute_step, ComputeStep.HEAD)
        self.assertEqual(relay.current_layer, 0)

    def test_replay_leaves_the_cache_alone(self):
        relay = make_relay(make_job())
        self.receive_and_compute(relay, 1)
        cache = relay.cache
        length = cache.get_seq_length()

        relay.receive_network_job(make_packet(relay, 1), "node-b")

        self.assertIs(relay.cache, cache)
        self.assertEqual(relay.cache.get_seq_length(), length)

    def test_replay_keeps_the_shared_kv_states_of_the_saved_pass(self):
        # Gemma 4 mutates `shared_kv_states` as the pass flows, so the next node
        # needs the copy this node produced, not the one it was handed.
        relay = make_relay(make_job())
        output = self.receive_and_compute(relay, 1)

        relay.receive_network_job(make_packet(relay, 1), "node-b")

        self.assertIs(relay.data.shared_kv_states, output.shared_kv_states)

    def test_next_pass_is_computed(self):
        relay = make_relay(make_job())
        self.receive_and_compute(relay, 1)

        incoming = make_packet(relay, 2)
        self.assertTrue(relay.receive_network_job(incoming, "node-b"))

        self.assertFalse(relay.passes.replaying)
        self.assertIs(relay.data, incoming.data)
        self.assertEqual(relay.compute_step, ComputeStep.LAYER)

    def test_second_visit_of_the_same_pass_is_computed(self):
        # A node hosting two layer ranges sees the same pass twice, once per
        # range. The second visit enters at a different layer, so it is not a
        # repeat of the first.
        relay = make_relay(make_job())
        self.receive_and_compute(relay, 1)

        incoming = make_packet(relay, 1, layer=4)
        self.assertTrue(relay.receive_network_job(incoming, "node-b"))

        self.assertFalse(relay.passes.replaying)
        self.assertIs(relay.data, incoming.data)

    def test_pass_out_of_sequence_is_refused(self):
        relay = make_relay(make_job())
        self.receive_and_compute(relay, 1)
        self.receive_and_compute(relay, 2)

        # Pass 3 never arrived, so the cache holds two positions, not three.
        self.assertFalse(relay.receive_network_job(make_packet(relay, 4), "node-b"))
        self.assertEqual(relay.passes.error, "pass out of sequence")

    def test_a_pass_already_left_behind_is_refused(self):
        relay = make_relay(make_job())
        self.receive_and_compute(relay, 1)
        self.receive_and_compute(relay, 2)

        self.assertFalse(relay.receive_network_job(make_packet(relay, 1), "node-b"))
        self.assertEqual(relay.passes.error, "pass out of sequence")

    def test_a_node_that_joins_the_job_late_accepts_the_pass(self):
        relay = make_relay(make_job())

        self.assertTrue(relay.receive_network_job(make_packet(relay, 9), "node-b"))

        self.assertFalse(relay.passes.replaying)
        self.assertIsNone(relay.passes.error)

    def test_a_peer_that_does_not_number_passes_is_always_computed(self):
        relay = make_relay(make_job())
        for _ in range(3):
            self.assertTrue(relay.receive_network_job(make_packet(relay, 0), "node-b"))
            self.assertFalse(relay.passes.replaying)
            compute_and_send(relay, make_data())

        self.assertIsNone(relay.passes.error)


class JobRestartTests(unittest.TestCase):
    """The origin's half: a bounced packet resends the pass in flight instead of
    embedding again."""

    def dispatch(self, origin: Job) -> JobData:
        """Run the origin through one pass, up to the handoff."""
        origin.passes.start()
        output = make_data(float(origin.passes.idx))
        origin.data = output
        origin.compute_step = ComputeStep.LAYER
        origin.current_layer = 2
        origin.passes.save(origin.data, origin.compute_step, origin.current_layer)
        origin.passes.sent()
        return output

    def bounce(self, origin: Job, pass_idx: int) -> NetworkJob:
        """The packet `JobReceiver.restart_token` sends back: no data, EMBED."""
        return NetworkJob(
            job_id=origin.job_id,
            pipe_id=origin.pipe_id,
            origin_node_id=origin.origin_node_id,
            current_layer=0,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.EMBED,
            times=[],
            pass_idx=pass_idx
        )

    def test_bounce_resends_the_pass_in_flight(self):
        origin = make_job()
        output = self.dispatch(origin)

        self.assertTrue(origin.receive_network_job(self.bounce(origin, origin.passes.idx), "node-a"))

        self.assertTrue(origin.passes.replaying)
        self.assertIs(origin.data, output)
        self.assertEqual(origin.compute_step, ComputeStep.LAYER)
        self.assertEqual(origin.current_layer, 2)

    def test_bounce_does_not_change_the_pass_number(self):
        origin = make_job()
        self.dispatch(origin)
        pass_idx = origin.passes.idx

        origin.receive_network_job(self.bounce(origin, pass_idx), "node-a")

        self.assertEqual(origin.passes.idx, pass_idx)
        self.assertEqual(origin.to_network_job().pass_idx, pass_idx)

    def test_bounce_from_a_peer_that_does_not_number_passes_is_honored(self):
        # Only one pass is ever in flight, so an unnumbered bounce can only be
        # about that pass.
        origin = make_job()
        output = self.dispatch(origin)

        self.assertTrue(origin.receive_network_job(self.bounce(origin, 0), "node-a"))

        self.assertIs(origin.data, output)

    def test_a_bounce_for_a_dead_pass_is_dropped(self):
        origin = make_job()
        self.dispatch(origin)
        self.dispatch(origin)

        self.assertFalse(origin.receive_network_job(self.bounce(origin, 1), "node-a"))
        self.assertIsNone(origin.passes.error)
        self.assertFalse(origin.passes.replaying)

    def test_retries_are_capped(self):
        origin = make_job()
        self.dispatch(origin)

        for _ in range(MAX_PASS_RETRIES):
            self.assertTrue(origin.receive_network_job(self.bounce(origin, origin.passes.idx), "node-a"))
            origin.passes.save(origin.data, origin.compute_step, origin.current_layer)
            origin.passes.sent()

        self.assertFalse(origin.receive_network_job(self.bounce(origin, origin.passes.idx), "node-a"))
        self.assertEqual(
            origin.passes.error,
            f"packet failed validation after {MAX_PASS_RETRIES} retries"
        )

    def test_the_retry_count_resets_with_the_next_pass(self):
        origin = make_job()
        self.dispatch(origin)
        origin.receive_network_job(self.bounce(origin, origin.passes.idx), "node-a")

        self.dispatch(origin)

        self.assertEqual(origin.passes.retries, 0)


class JobAttemptTests(unittest.TestCase):
    """`attempt` is what carries the correctness of a rebuild. The origin's
    ABORT travels on a different connection from its own retry and can lose the
    race, so the retry itself has to say that everything held here is stale."""

    def relay_holding_a_pass(self) -> Job:
        relay = make_relay(make_job())
        relay.receive_network_job(make_packet(relay, 1), "node-b")
        compute_and_send(relay, make_data(1.0))
        relay.caching.prefix_len = 384
        return relay

    def test_a_higher_attempt_throws_the_cache_and_the_numbering_away(self):
        relay = self.relay_holding_a_pass()
        packet = make_packet(relay, 1)
        packet.attempt = 1

        self.assertTrue(relay.receive_network_job(packet, "node-b"))

        self.assertEqual(relay.cache.get_seq_length(), 0)
        self.assertEqual(relay.passes.attempt, 1)
        self.assertEqual(relay.passes.outputs, {})
        self.assertEqual(relay.caching.prefix_len, 0)
        self.assertFalse(relay.passes.replaying)

    def test_a_lower_attempt_is_dropped_without_canceling(self):
        relay = self.relay_holding_a_pass()
        relay.passes.reset(2)
        packet = make_packet(relay, 1)
        packet.attempt = 1

        self.assertFalse(relay.receive_network_job(packet, "node-b"))
        self.assertIsNone(relay.passes.error)

    def test_the_same_attempt_is_processed_as_before(self):
        relay = self.relay_holding_a_pass()

        self.assertTrue(relay.receive_network_job(make_packet(relay, 1), "node-b"))

        # Same pass, same entry point: a replay, not a rebuild.
        self.assertTrue(relay.passes.replaying)
        self.assertEqual(relay.cache.get_seq_length(), 1)

    def test_the_attempt_goes_on_the_wire(self):
        origin = make_job()
        origin.passes.reset(3)

        self.assertEqual(origin.to_network_job().attempt, 3)

    def test_a_rebuild_starts_the_job_over_from_tokenize(self):
        origin = make_job()
        origin.prompt_tokens = 600
        origin.input_ids = list(range(640))
        origin.current_token = 40
        origin.caching.prefix_len = 384
        origin.caching.write_points = [512]
        fill_cache(origin)
        origin.passes.start()

        origin.rebuild()

        self.assertEqual(origin.passes.attempt, 1)
        self.assertEqual(origin.passes.idx, 0)
        self.assertEqual(origin.compute_step, ComputeStep.TOKENIZE)
        self.assertEqual(origin.prompt_tokens, 0)
        self.assertEqual(origin.input_ids, [])
        self.assertEqual(origin.current_token, 0)
        self.assertEqual(origin.cache.get_seq_length(), 0)
        self.assertEqual(origin.caching.prefix_len, 0)
        self.assertEqual(origin.caching.write_points, [])
        self.assertFalse(origin.caching.options.enabled)


class JobCacheTagTests(unittest.TestCase):
    """A layer node never sees a token, so the write boundary has to be told to
    it, and it rides the packet that already visits every node in the pass."""

    def test_the_write_tag_is_taken_off_the_packet(self):
        relay = make_relay(make_job())
        packet = make_packet(relay, 1)
        packet.cache_write_id = b"\x07" * 32
        packet.cache_write_tokens = 512

        relay.receive_network_job(packet, "node-b")

        self.assertEqual(relay.caching.pending_write_id, b"\x07" * 32)
        self.assertEqual(relay.caching.pending_write_tokens, 512)

    def test_an_untagged_packet_clears_the_tag_from_the_pass_before(self):
        relay = make_relay(make_job())
        relay.caching.pending_write_id = b"\x07" * 32
        relay.caching.pending_write_tokens = 512

        relay.receive_network_job(make_packet(relay, 1), "node-b")

        self.assertIsNone(relay.caching.pending_write_id)
        self.assertEqual(relay.caching.pending_write_tokens, 0)

    def test_the_read_tags_are_carried_on_to_the_next_node(self):
        relay = make_relay(make_job())
        packet = make_packet(relay, 1)
        packet.cache_use_id = b"\x08" * 32
        packet.cache_use_tokens = 384
        packet.cache_reserve_tokens = 2000

        relay.receive_network_job(packet, "node-b")
        onward = relay.to_network_job()

        self.assertEqual(onward.cache_use_id, b"\x08" * 32)
        self.assertEqual(onward.cache_use_tokens, 384)
        self.assertEqual(onward.cache_reserve_tokens, 2000)

    def test_the_read_tags_ride_only_the_first_pass(self):
        relay = make_relay(make_job())
        packet = make_packet(relay, 1)
        packet.cache_use_id = b"\x08" * 32
        packet.cache_use_tokens = 384
        relay.receive_network_job(packet, "node-b")
        compute_and_send(relay, make_data())
        # The second pass: every node already has its job.
        relay.receive_network_job(make_packet(relay, 2, layer=1), "node-b")

        onward = relay.to_network_job()

        self.assertEqual(onward.cache_use_id, b"")
        self.assertEqual(onward.cache_use_tokens, 0)

    def test_a_replayed_first_pass_still_carries_the_read_tags(self):
        """The node that bounced the packet has not created its job yet, so the
        resend is its only chance to be told what to adopt."""
        origin = make_job()
        origin.caching.use_id = b"\x08" * 32
        origin.caching.use_tokens = 384
        origin.caching.reserve_tokens = 2000
        origin.passes.start()
        origin.passes.save(origin.data, origin.compute_step, origin.current_layer)

        saved = origin.passes.restart(origin.passes.idx)
        assert saved is not None
        origin.replay(saved)
        onward = origin.to_network_job()

        self.assertEqual(onward.cache_use_id, b"\x08" * 32)
        self.assertEqual(onward.cache_use_tokens, 384)
        self.assertEqual(onward.cache_reserve_tokens, 2000)

    def test_the_origin_does_not_take_tags_off_its_own_returning_packet(self):
        origin = make_job()
        packet = make_packet(origin, 1, step=ComputeStep.HEAD)
        packet.cache_write_id = b"\x07" * 32
        packet.cache_write_tokens = 512

        origin.receive_network_job(packet, origin.origin_node_id)

        self.assertIsNone(origin.caching.pending_write_id)


if __name__ == "__main__":
    unittest.main()
