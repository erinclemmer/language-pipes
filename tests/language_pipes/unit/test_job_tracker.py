import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_progress import JobProgress
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.enums import ComputeStep, JobStatus


def make_tracker() -> JobTracker:
    # The stale-job sweeper thread is not under test
    with patch("language_pipes.jobs.job_tracker.Thread"):
        return JobTracker()


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

        job = tracker.add_job(make_network_job(chunk_width=8), PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]

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

        job = tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]
        assert job is not None
        job.receive_network_job(network_job, "node-b")

        self.assertEqual(job.display_progress().prompt_tokens, 200)
        self.assertEqual(job.display_progress().prefill_tokens, 64)

    def test_rejects_a_job_it_already_tracks(self):
        tracker = make_tracker()
        config = PretrainedConfig(num_hidden_layers=1) # pyright: ignore[reportCallIssue]

        self.assertIsNotNone(tracker.add_job(make_network_job(), config))
        self.assertIsNone(tracker.add_job(make_network_job(), config))

    def test_rejects_a_job_that_was_never_embedded(self):
        tracker = make_tracker()
        network_job = make_network_job()
        network_job.data.state = None  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        with self.assertRaises(Exception):  # noqa: B017
            tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1)) # pyright: ignore[reportCallIssue]


if __name__ == "__main__":
    unittest.main()
