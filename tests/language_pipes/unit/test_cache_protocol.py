"""The prompt cache's back-channel: what a node says when it cannot adopt or
cannot store, and what the origin does about it."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import PretrainedConfig

from language_pipes.jobs.cache_packets import CacheReason, CacheStatus
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_receiver import CACHE_PROTOCOL, JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, PromptCache
from language_pipes.util.byte_helper import ByteHelper
from language_pipes.util.enums import ComputeStep

from util import FakeEndModel, FakeModel, PipeWrapper, make_job_data

NUM_LAYERS = 2


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
    def __init__(self, pipe, router: FakeRouter):
        self.pipe = pipe
        self.router_pipes = type("FakeRouterPipes", (), {"router": router})()

    def get_pipe_by_pipe_id(self, pipe_id: str):
        return self.pipe if pipe_id == self.pipe.pipe_id else None


class FakeModelManager:
    def __init__(self, end_model=None):
        self.end_model = end_model

    def get_end_model(self, model_id: str):
        return self.end_model

    def get_config(self, model_id: str):
        return PretrainedConfig(num_hidden_layers=NUM_LAYERS)  # pyright: ignore[reportCallIssue]


def make_receiver(node_id: str, pipe, end_model=None, prompt_cache=None):
    router = FakeRouter(node_id)
    with patch("language_pipes.jobs.job_tracker.Thread"):
        tracker = JobTracker(prompt_cache)
    tracker.shutdown = True
    receiver = JobReceiver(
        job_factory=None,  # pyright: ignore[reportArgumentType]
        job_tracker=tracker,
        pipe_manager=FakePipeManager(pipe, router),  # pyright: ignore[reportArgumentType]
        model_manager=FakeModelManager(end_model),  # pyright: ignore[reportArgumentType]
        is_shutdown=lambda: True,
        get_max_node_jobs=lambda: 10,
    )
    return receiver, tracker, router


def read_status(data: bytes) -> CacheStatus:
    bts = ByteHelper(data)
    assert bts.read_int() == CACHE_PROTOCOL
    return CacheStatus.from_bytes(bts.read_bytes())


def statuses(router: FakeRouter):
    return [(node_id, read_status(data)) for node_id, data in router.sent]


class CacheStatusPacketTests(unittest.TestCase):
    def test_round_trips(self):
        status = CacheStatus("job-1", "pipe-1", 3, CacheReason.NO_STORE)

        parsed = CacheStatus.from_bytes(status.to_bytes())

        self.assertEqual(parsed.job_id, "job-1")
        self.assertEqual(parsed.pipe_id, "pipe-1")
        self.assertEqual(parsed.attempt, 3)
        self.assertEqual(parsed.reason, CacheReason.NO_STORE)


class LayerNodeAnswersTests(unittest.TestCase):
    """The node that cannot adopt is the one that has to say so, and it must
    not compute the pass it was sent."""

    def setUp(self):
        self.cache = PromptCache(lambda: 300, lambda: 100000)
        self.segment = FakeModel("node-2", 1, 1, num_hidden_layers=NUM_LAYERS)
        self.pipe = PipeWrapper("node-2", "model-1", [self.segment])
        self.receiver, self.tracker, self.router = make_receiver(
            "node-2", self.pipe, prompt_cache=self.cache
        )

    def packet(self, attempt: int = 0) -> NetworkJob:
        data = make_job_data()
        data.state = torch.zeros((1, 1))
        return NetworkJob(
            job_id="job-1",
            pipe_id=self.pipe.pipe_id,
            origin_node_id="node-1",
            current_layer=1,
            data=data,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
            pass_idx=1,
            attempt=attempt,
            cache_use_id=b"\x05" * 32,
            cache_use_tokens=BLOCK_SIZE * 3,
            cache_reserve_tokens=2000
        )

    def test_a_node_that_does_not_hold_the_prefix_reports_a_miss(self):
        self.receiver._process_network_job(self.packet(attempt=2))

        self.assertEqual(len(self.router.sent), 1)
        node_id, status = statuses(self.router)[0]
        self.assertEqual(node_id, "node-1")
        self.assertEqual(status.reason, CacheReason.MISS)
        self.assertEqual(status.attempt, 2)
        self.assertEqual(status.job_id, "job-1")

    def test_a_node_that_misses_does_not_compute_the_pass(self):
        self.receiver._process_network_job(self.packet())

        self.assertIsNone(self.tracker.get_job("job-1"))
        self.assertEqual(self.pipe.sent_jobs, [])

    def test_a_node_with_no_budget_reports_it_and_runs_the_job_anyway(self):
        self.cache = PromptCache(lambda: 300, lambda: 100)
        self.receiver, self.tracker, self.router = make_receiver(
            "node-2", self.pipe, prompt_cache=self.cache
        )
        packet = self.packet()
        packet.cache_use_id = b""
        packet.cache_use_tokens = 0

        self.receiver._process_network_job(packet)

        self.assertEqual([s.reason for _, s in statuses(self.router)], [CacheReason.NO_STORE])
        self.assertEqual(len(self.pipe.sent_jobs), 1)


class OriginRebuildTests(unittest.TestCase):
    """A miss anywhere means every cache on the pipe is thrown away and the
    prefill starts over, with reuse off for the rest of the job."""

    def setUp(self):
        self.cache = PromptCache(lambda: 300, lambda: 100000)
        self.end_model = FakeEndModel(num_local_layers=1)
        self.pipe = PipeWrapper("node-1", "model-1", [
            FakeModel("node-2", 1, 1, virtual=True, num_hidden_layers=NUM_LAYERS)
        ])
        self.receiver, self.tracker, self.router = make_receiver(
            "node-1", self.pipe, self.end_model, self.cache
        )
        self.job = self.dispatched_job()

    def dispatched_job(self) -> Job:
        job = Job(
            origin_node_id="node-1",
            messages=[],
            pipe_id=self.pipe.pipe_id,
            model_id="model-1",
            config=PretrainedConfig(num_hidden_layers=NUM_LAYERS),  # pyright: ignore[reportCallIssue]
        )
        job.job_id = "job-1"
        job.prompt_tokens = 600
        job.input_ids = list(range(600))
        job.caching.options.enabled = True
        job.caching.scope = self.cache.scope("node-1", "api-key", "tenant")
        job.caching.ids = self.cache.chain(job.caching.scope, job.input_ids)
        job.caching.prefix_len = BLOCK_SIZE * 3
        job.caching.cached_tokens = BLOCK_SIZE * 3
        job.caching.write_points = [BLOCK_SIZE * 4]
        job.caching.use_id = job.caching.ids[3]
        job.caching.use_tokens = BLOCK_SIZE * 3
        job.caching.reserve_tokens = 2000
        job.caching.reserved = True
        self.cache.reserve("job-1", 2000)
        job.init_chunking()
        job.passes.start()
        self.tracker.jobs_pending["key"] = [job]
        return job

    def miss(self, attempt: int = 0):
        self.receiver.receive_cache_status(
            "node-2",
            CacheStatus("job-1", self.pipe.pipe_id, attempt, CacheReason.MISS).to_bytes()
        )

    def test_a_miss_restarts_the_job_with_reuse_off(self):
        self.miss()

        self.assertEqual(self.job.passes.attempt, 1)
        self.assertEqual(self.job.caching.prefix_len, 0)
        self.assertEqual(self.job.caching.cached_tokens, 0)
        self.assertEqual(self.job.caching.write_points, [])
        self.assertFalse(self.job.caching.options.enabled)

    def test_the_retry_names_no_prefix_and_carries_the_new_attempt(self):
        self.miss()

        self.assertEqual(len(self.pipe.sent_jobs), 1)
        sent = self.pipe.sent_jobs[0]
        self.assertEqual(sent.attempt, 1)
        self.assertEqual(sent.cache_use_id, b"")
        self.assertEqual(sent.cache_use_tokens, 0)
        self.assertEqual(sent.cache_reserve_tokens, 0)
        self.assertEqual(sent.cache_write_id, b"")

    def test_the_retry_prefills_from_the_first_token(self):
        self.miss()

        # The fake tokenizer produces a two-token prompt, embedded in one pass
        # from position 0 - nothing of the adopted prefix survives.
        self.assertEqual(self.job.past_seen_tokens(), 0)
        self.assertEqual(self.job.chunking.start_offset, 0)

    def test_every_other_node_on_the_pipe_is_told_to_drop_the_job(self):
        self.miss()

        sent = statuses(self.router)
        self.assertEqual([node_id for node_id, _ in sent], ["node-2"])
        self.assertEqual(sent[0][1].reason, CacheReason.ABORT)
        # The ABORT names the attempt it kills, not the one replacing it.
        self.assertEqual(sent[0][1].attempt, 0)

    def test_the_origin_gives_its_reservation_back(self):
        self.miss()

        self.assertFalse(self.job.caching.reserved)
        self.assertEqual(self.cache.used_tokens(), 0)

    def test_a_miss_naming_a_dead_attempt_does_not_restart_the_retry(self):
        self.miss()
        self.pipe.sent_jobs = []
        self.router.sent = []

        # A second node's answer to the attempt that was already abandoned.
        self.miss(attempt=0)

        self.assertEqual(self.job.passes.attempt, 1)
        self.assertEqual(self.pipe.sent_jobs, [])
        self.assertEqual(self.router.sent, [])

    def test_no_store_stops_the_job_writing_anything(self):
        self.receiver.receive_cache_status(
            "node-2",
            CacheStatus("job-1", self.pipe.pipe_id, 0, CacheReason.NO_STORE).to_bytes()
        )

        self.assertEqual(self.job.caching.write_points, [])
        self.assertIsNone(self.job.caching.pending_write_id)
        self.assertTrue(self.job.caching.writes_stopped)
        # Not a restart: the job keeps the prefix it adopted and runs on.
        self.assertEqual(self.job.caching.prefix_len, BLOCK_SIZE * 3)
        self.assertEqual(self.pipe.sent_jobs, [])

    def test_a_status_for_an_unknown_job_is_ignored(self):
        self.receiver.receive_cache_status(
            "node-2",
            CacheStatus("job-9", self.pipe.pipe_id, 0, CacheReason.MISS).to_bytes()
        )

        self.assertEqual(self.job.passes.attempt, 0)

    def test_an_unparseable_status_is_ignored(self):
        self.receiver.receive_cache_status("node-2", b"not a cache status")

        self.assertEqual(self.job.passes.attempt, 0)

    def test_the_origin_ignores_an_abort_for_its_own_job(self):
        self.receiver.receive_cache_status(
            "node-2",
            CacheStatus("job-1", self.pipe.pipe_id, 0, CacheReason.ABORT).to_bytes()
        )

        self.assertIsNotNone(self.tracker.get_job("job-1"))


class AbortTests(unittest.TestCase):
    """`ABORT` frees a stale cache now instead of at the next packet. It is an
    optimization, so it is dropped whenever it might do harm."""

    def setUp(self):
        self.cache = PromptCache(lambda: 300, lambda: 100000)
        self.segment = FakeModel("node-2", 1, 1, num_hidden_layers=NUM_LAYERS)
        self.pipe = PipeWrapper("node-2", "model-1", [self.segment])
        self.receiver, self.tracker, self.router = make_receiver(
            "node-2", self.pipe, prompt_cache=self.cache
        )
        self.job = Job(
            origin_node_id="node-1",
            messages=[],
            pipe_id=self.pipe.pipe_id,
            model_id="model-1",
            config=PretrainedConfig(num_hidden_layers=NUM_LAYERS),  # pyright: ignore[reportCallIssue]
        )
        self.job.job_id = "job-1"
        self.tracker.jobs_pending["network"] = [self.job]
        self.cache.reserve("job-1", 2000)
        self.job.caching.reserved = True

    def abort(self, attempt: int):
        self.receiver.receive_cache_status(
            "node-1",
            CacheStatus("job-1", self.pipe.pipe_id, attempt, CacheReason.ABORT).to_bytes()
        )

    def test_an_abort_for_the_current_attempt_drops_the_job(self):
        self.abort(0)

        self.assertIsNone(self.tracker.get_job("job-1"))
        self.assertEqual(self.cache.used_tokens(), 0)

    def test_an_abort_that_lost_the_race_against_the_retry_is_ignored(self):
        # The retry arrived first and moved this node to attempt 1; the ABORT
        # for attempt 0 must not take the retry's job away with it.
        self.job.passes.reset(1)

        self.abort(0)

        self.assertIsNotNone(self.tracker.get_job("job-1"))

    def test_a_layer_node_does_not_act_on_a_miss_for_someone_elses_job(self):
        """`MISS` travels to the origin; a node hosting layers has no job of its
        own to restart and must not try."""
        self.receiver.receive_cache_status(
            "node-1",
            CacheStatus("job-1", self.pipe.pipe_id, 0, CacheReason.MISS).to_bytes()
        )

        self.assertIsNotNone(self.tracker.get_job("job-1"))
        self.assertEqual(self.job.passes.attempt, 0)
        self.assertEqual(self.pipe.sent_jobs, [])

    def test_queued_packets_for_the_aborted_job_are_discarded(self):
        network_job = NetworkJob(
            job_id="job-1",
            pipe_id=self.pipe.pipe_id,
            origin_node_id="node-1",
            current_layer=1,
            data=None,
            data_hash=b"",
            compute_step=ComputeStep.LAYER,
            times=[],
        )
        self.receiver.receive_data("node-1", network_job.to_bytes())

        self.abort(0)

        self.assertEqual(self.receiver.job_queue, {})


if __name__ == "__main__":
    unittest.main()
