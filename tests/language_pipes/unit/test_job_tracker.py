import os
import sys
import unittest
from time import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.cache_policy import CacheOutcome, CachePolicy
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_progress import JobProgress
from language_pipes.jobs.job_tracker import EXPIRED_JOB_TIME, JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, PromptCache
from language_pipes.util.enums import ComputeStep, JobStatus


def make_tracker(prompt_cache: PromptCache | None = None) -> JobTracker:
    # The stale-job sweeper thread is not under test
    with patch("language_pipes.jobs.job_tracker.Thread"):
        return JobTracker(prompt_cache)


def make_job(job_id: str = "job-1", **kwargs) -> Job:
    defaults = {
        "origin_node_id": "node-a",
        "messages": [],
        "pipe_id": "pipe-1",
        "model_id": "model-1",
        "config": PretrainedConfig(num_hidden_layers=1), # pyright: ignore[reportCallIssue]
    }
    defaults.update(kwargs)
    job = Job(**defaults)
    job.job_id = job_id
    return job


def make_network_job(chunk_width: int = 8, progress: JobProgress | None = None) -> NetworkJob:
    return NetworkJob(
        job_id="job-1",
        pipe_id="pipe-1",
        origin_node_id="node-a",
        current_layer=0,
        data=JobData(
            state=torch.zeros((1, chunk_width, 4)),
            cache_position=torch.tensor([]),
            position_ids=torch.tensor([]),
            causal_mask={},
            position_embeddings={}
        ),
        data_hash=b"",
        compute_step=ComputeStep.LAYER,
        times=[],
        progress=progress
    )


class CancelJobTests(unittest.TestCase):
    def test_cancel_resolves_waiting_caller(self):
        tracker = make_tracker()
        resolved = []
        job = make_job(resolve=lambda j: resolved.append(j))
        tracker.jobs_pending["key-1"] = [job]

        tracker.cancel_job(job, "layers for model-1 unloaded")

        self.assertEqual(resolved, [job])

    def test_cancel_records_reason_and_status(self):
        tracker = make_tracker()
        job = make_job()
        tracker.jobs_pending["key-1"] = [job]

        tracker.cancel_job(job, "end model for model-1 unloaded")

        self.assertEqual(job.cancel_reason, "end model for model-1 unloaded")
        self.assertEqual(job.status, JobStatus.ERROR)
        self.assertTrue(job.stale)

    def test_cancel_removes_job_from_pending(self):
        tracker = make_tracker()
        job = make_job()
        tracker.jobs_pending["key-1"] = [job]

        tracker.cancel_job(job, "unloaded")

        self.assertEqual(tracker.jobs_pending["key-1"], [])
        self.assertIsNone(tracker.get_job("job-1"))

    def test_cancel_is_a_no_op_once_completed(self):
        tracker = make_tracker()
        resolved = []
        job = make_job(resolve=lambda j: resolved.append(j))
        tracker.jobs_pending["key-1"] = [job]
        tracker.complete_job(job)

        tracker.cancel_job(job, "unloaded")

        self.assertEqual(len(resolved), 1)
        self.assertIsNone(job.cancel_reason)

    def test_completing_job_without_resolve_clears_it_from_pending(self):
        # Jobs forwarded from another node have no promise to resolve, but they
        # still have to leave the pending list rather than wait for the timeout.
        tracker = make_tracker()
        job = make_job()
        tracker.jobs_pending["network"] = [job]

        tracker.complete_job(job)

        self.assertEqual(tracker.jobs_pending["network"], [])


class JobLookupTests(unittest.TestCase):
    def test_jobs_for_pipes_matches_only_listed_pipes(self):
        tracker = make_tracker()
        on_pipe = make_job("job-1", pipe_id="pipe-1")
        off_pipe = make_job("job-2", pipe_id="pipe-2")
        tracker.jobs_pending["network"] = [on_pipe, off_pipe]

        self.assertEqual(tracker.jobs_for_pipes(["pipe-1"]), [on_pipe])

    def test_jobs_for_model_can_filter_by_origin(self):
        tracker = make_tracker()
        ours = make_job("job-1", origin_node_id="node-a")
        theirs = make_job("job-2", origin_node_id="node-b")
        tracker.jobs_pending["network"] = [ours, theirs]

        self.assertEqual(tracker.jobs_for_model("model-1", "node-a"), [ours])
        self.assertEqual(len(tracker.jobs_for_model("model-1")), 2)

    def test_jobs_for_model_ignores_other_models(self):
        tracker = make_tracker()
        job = make_job("job-1", model_id="model-2")
        tracker.jobs_pending["network"] = [job]

        self.assertEqual(tracker.jobs_for_model("model-1"), [])


class AddJobTests(unittest.TestCase):
    """`add_job` builds the local record for a job this node only hosts layers
    for, so anything the origin owns has to stay unset here."""

    def test_prompt_tokens_is_not_guessed_from_the_state_in_flight(self):
        tracker = make_tracker()

        job, _ = tracker.add_job(make_network_job(chunk_width=8), PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]

        assert job is not None
        # The state is one pass wide - a decode token or a prefill chunk - and
        # says nothing about how long the prompt is
        self.assertEqual(job.prompt_tokens, 0)

    def test_prompt_tokens_comes_from_the_origin_instead(self):
        tracker = make_tracker()
        network_job = make_network_job(
            chunk_width=8,
            progress=JobProgress(
                current_token=0,
                prompt_tokens=200,
                prefilling=True,
                prefill_tokens=64
            )
        )

        job, _ = tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]
        assert job is not None
        job.receive_network_job(network_job, "node-b")

        self.assertEqual(job.display_progress().prompt_tokens, 200)
        self.assertEqual(job.display_progress().prefill_tokens, 64)

    def test_rejects_a_job_it_already_tracks(self):
        tracker = make_tracker()
        config = PretrainedConfig(num_hidden_layers=1) # pyright: ignore[reportCallIssue]

        self.assertIsNotNone(tracker.add_job(make_network_job(), config)[0])
        self.assertIsNone(tracker.add_job(make_network_job(), config)[0])

    def test_rejects_a_job_that_was_never_embedded(self):
        tracker = make_tracker()
        network_job = make_network_job()
        network_job.data.state = None  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        with self.assertRaises(Exception):  # noqa: B017
            tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]


class FakeSegment:
    def __init__(self, node_id, start_layer, end_layer, process_id, virtual=False):
        self.node_id = node_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.process_id = process_id
        self.virtual = virtual
        self.loaded = True
        self.num_hidden_layers = 4


class FakePipe:
    """Just enough of a pipe for `CachePolicy` to name a node's slice."""

    def __init__(self, segments):
        self.model_id = "model-1"
        self.pipe_id = "pipe-1"
        self.segments = segments

    def num_hidden_layers(self):
        return 4

    def get_layer(self, layer, need_physical=False):
        for s in self.segments:
            if s.start_layer <= layer <= s.end_layer and (not need_physical or not s.virtual):
                return s
        return None


class LayerNodeReadPathTests(unittest.TestCase):
    """A layer node's whole read path runs as its `Job` is created: it never
    sees a token, so the origin names one entry and it either holds it or does
    not."""

    def setUp(self):
        self.cache = PromptCache(lambda: 300, lambda: 100000)
        self.tracker = make_tracker(self.cache)
        self.pipe = FakePipe([
            FakeSegment("node-a", 0, 1, "proc-a", virtual=True),
            FakeSegment("node-b", 2, 3, "proc-b"),
        ])
        self.policy = CachePolicy("node-b", self.pipe, None, self.cache)  # pyright: ignore[reportArgumentType]
        self.cache_id = b"\x05" * 32

    def seed_entry(self, tokens: int = BLOCK_SIZE * 3, origin: str = "node-a"):
        seeded = DynamicCache(config=PretrainedConfig(num_hidden_layers=4)) # pyright: ignore[reportCallIssue]
        seeded.update(torch.ones(1, 1, tokens, 4), torch.ones(1, 1, tokens, 4), 2)
        self.cache.store(
            self.cache_id, seeded, origin, "model-1", ["proc-b"], 2, 3, tokens
        )

    def packet(self, use_tokens: int = BLOCK_SIZE * 3, reserve: int = 2000) -> NetworkJob:
        network_job = make_network_job()
        network_job.cache_use_id = self.cache_id if use_tokens > 0 else b""
        network_job.cache_use_tokens = use_tokens
        network_job.cache_reserve_tokens = reserve
        return network_job

    def add(self, network_job: NetworkJob):
        return self.tracker.add_job(
            network_job, PretrainedConfig(num_hidden_layers=4), "model-1", self.policy # pyright: ignore[reportCallIssue]
        )

    def test_a_hit_adopts_the_entry_into_the_new_job(self):
        self.seed_entry()

        job, outcome = self.add(self.packet())

        assert job is not None
        self.assertEqual(outcome, CacheOutcome.OK)
        self.assertEqual(job.caching.prefix_len, BLOCK_SIZE * 3)
        self.assertEqual(job.cache.get_seq_length(2), BLOCK_SIZE * 3)

    def test_a_miss_adds_no_job_at_all(self):
        job, outcome = self.add(self.packet())

        self.assertIsNone(job)
        self.assertEqual(outcome, CacheOutcome.MISS)
        self.assertIsNone(self.tracker.get_job("job-1"))

    def test_an_entry_stored_for_another_origin_is_a_miss(self):
        self.seed_entry(origin="node-z")

        job, outcome = self.add(self.packet())

        self.assertIsNone(job)
        self.assertEqual(outcome, CacheOutcome.MISS)

    def test_an_entry_of_a_different_length_than_the_origin_says_is_a_miss(self):
        """The count is what the job resumes at; disagreeing about it would put
        every later token at the wrong position."""
        self.seed_entry(tokens=BLOCK_SIZE * 2)

        job, outcome = self.add(self.packet(use_tokens=BLOCK_SIZE * 3))

        self.assertIsNone(job)
        self.assertEqual(outcome, CacheOutcome.MISS)

    def test_a_node_hosting_none_of_this_pipe_is_a_miss(self):
        self.seed_entry()
        policy = CachePolicy("node-c", self.pipe, None, self.cache)  # pyright: ignore[reportArgumentType]

        job, outcome = self.tracker.add_job(
            self.packet(), PretrainedConfig(num_hidden_layers=4), "model-1", policy # pyright: ignore[reportCallIssue]
        )

        self.assertIsNone(job)
        self.assertEqual(outcome, CacheOutcome.MISS)

    def test_a_refused_reservation_reports_no_store_but_still_runs_the_job(self):
        self.cache = PromptCache(lambda: 300, lambda: 100)
        self.tracker = make_tracker(self.cache)
        self.policy = CachePolicy("node-b", self.pipe, None, self.cache)  # pyright: ignore[reportArgumentType]

        job, outcome = self.add(self.packet(use_tokens=0))

        self.assertIsNotNone(job)
        self.assertEqual(outcome, CacheOutcome.NO_STORE)
        self.assertIsNotNone(self.tracker.get_job("job-1"))

    def test_a_packet_with_no_tags_touches_the_cache_at_all(self):
        job, outcome = self.add(make_network_job())

        self.assertIsNotNone(job)
        self.assertEqual(outcome, CacheOutcome.OK)
        self.assertEqual(self.cache.used_tokens(), 0)
        self.assertEqual(self.cache.stats().hits + self.cache.stats().misses, 0)

    def test_a_device_resident_local_entry_reserves_the_wire_amount_unchanged(self):
        """The wire number is the origin's own estimate; a node whose local
        copy of the entry is device-resident owes exactly that (cache_tier_plan.md §5.5)."""
        self.seed_entry()

        job, outcome = self.add(self.packet(reserve=2000))

        assert job is not None
        self.assertEqual(outcome, CacheOutcome.OK)
        self.assertEqual(self.cache._reservations[job.job_id].tokens, 2000)

    def test_a_host_resident_local_entry_subtracts_its_own_prefix(self):
        """Residency is local to each node: this node's own copy of the entry
        being host-resident means it does not pay the origin's device-pressure
        charge, even though the wire number assumed it would."""
        self.seed_entry()
        self.cache._entries[self.cache_id].on_device = False

        job, outcome = self.add(self.packet(reserve=2000))

        assert job is not None
        self.assertEqual(outcome, CacheOutcome.OK)
        self.assertEqual(
            self.cache._reservations[job.job_id].tokens, 2000 - BLOCK_SIZE * 3
        )

    def test_the_subtraction_never_goes_negative(self):
        self.seed_entry()
        self.cache._entries[self.cache_id].on_device = False

        job, outcome = self.add(self.packet(reserve=BLOCK_SIZE))

        assert job is not None
        self.assertEqual(outcome, CacheOutcome.OK)
        self.assertEqual(self.cache._reservations[job.job_id].tokens, 0)


class CacheReservationTests(unittest.TestCase):
    """A dropped connection or an expired job must not leak cache budget."""

    def setUp(self):
        self.prompt_cache = PromptCache(lambda: 300, lambda: 10000)
        self.tracker = make_tracker(self.prompt_cache)

    def track(self, job: Job):
        self.tracker.jobs_pending["key"] = [job]
        self.prompt_cache.reserve(job.job_id, 500)

    def run_one_sweep(self):
        """`check_stale_jobs` loops forever; stop it at its first sleep."""
        def stop(_seconds):
            self.tracker.shutdown = True
        with patch("language_pipes.jobs.job_tracker.sleep", side_effect=stop):
            self.tracker.check_stale_jobs()

    def test_completing_a_job_releases_its_reservation(self):
        job = make_job()
        self.track(job)

        self.tracker.complete_job(job)

        self.assertEqual(self.prompt_cache.used_tokens(), 0)

    def test_canceling_a_job_releases_its_reservation(self):
        job = make_job()
        self.track(job)

        self.tracker.cancel_job(job, "layers unloaded")

        self.assertEqual(self.prompt_cache.used_tokens(), 0)

    def test_a_job_reaped_as_stale_releases_its_reservation(self):
        job = make_job()
        self.track(job)
        job.last_update = time() - EXPIRED_JOB_TIME - 1

        self.run_one_sweep()

        self.assertIsNone(self.tracker.get_job(job.job_id))
        self.assertEqual(self.prompt_cache.used_tokens(), 0)

    def test_the_stale_sweep_also_expires_cache_entries(self):
        self.prompt_cache.store(
            b"a" * 32, DynamicCache(config=PretrainedConfig(num_hidden_layers=1)), # pyright: ignore[reportCallIssue]
            "node-a", "model-1", ["proc"], 0, 0, 256
        )
        self.prompt_cache._entries[b"a" * 32].expires_at = time() - 1

        self.run_one_sweep()

        self.assertEqual(self.prompt_cache.stats().entries, 0)

    def test_a_tracker_without_a_cache_behaves_as_before(self):
        tracker = make_tracker()
        job = make_job()
        tracker.jobs_pending["key"] = [job]

        tracker.complete_job(job)

        self.assertIsNone(tracker.get_job(job.job_id))


class DemotionOnJobEndTests(unittest.TestCase):
    """Host tiering (cache_tier_plan.md §5.3): nothing still needs an entry on
    the device once the job that touched it has ended, so ending a job is what
    queues its entries for demotion."""

    def setUp(self):
        self.prompt_cache = PromptCache(lambda: 300, lambda: 100000, lambda: 100000)
        self.tracker = make_tracker(self.prompt_cache)

    def job_with_touched(self, *touched_ids: bytes) -> Job:
        job = make_job()
        job.caching.touched_ids = list(touched_ids)
        return job

    def test_remove_job_queues_its_touched_entries_for_demotion(self):
        job = self.job_with_touched(b"a" * 32, b"b" * 32)
        self.tracker.jobs_pending["key"] = [job]

        self.tracker.remove_job(job.job_id)

        self.assertEqual(self.prompt_cache._demote_queue.qsize(), 2)

    def test_completing_a_job_reaches_the_same_path(self):
        job = self.job_with_touched(b"a" * 32)
        self.tracker.jobs_pending["key"] = [job]

        self.tracker.complete_job(job)

        self.assertEqual(self.prompt_cache._demote_queue.qsize(), 1)

    def test_canceling_a_job_reaches_the_same_path(self):
        job = self.job_with_touched(b"a" * 32)
        self.tracker.jobs_pending["key"] = [job]

        self.tracker.cancel_job(job, "reason")

        self.assertEqual(self.prompt_cache._demote_queue.qsize(), 1)

    def test_a_job_that_touched_nothing_queues_nothing(self):
        job = self.job_with_touched()
        self.tracker.jobs_pending["key"] = [job]

        self.tracker.remove_job(job.job_id)

        self.assertEqual(self.prompt_cache._demote_queue.qsize(), 0)

    def test_a_job_still_pending_queues_nothing(self):
        """Only ending the job triggers this - a job mid-decode must keep its
        entry device-resident."""
        job = self.job_with_touched(b"a" * 32)
        self.tracker.jobs_pending["key"] = [job]

        self.assertEqual(self.prompt_cache._demote_queue.qsize(), 0)

    def test_a_tracker_without_a_cache_tolerates_touched_ids(self):
        tracker = make_tracker()
        job = self.job_with_touched(b"a" * 32)
        tracker.jobs_pending["key"] = [job]

        tracker.remove_job(job.job_id)  # must not raise

        self.assertIsNone(tracker.get_job(job.job_id))


if __name__ == "__main__":
    unittest.main()
