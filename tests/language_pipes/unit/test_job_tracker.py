import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
from transformers import PretrainedConfig

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_progress import JobProgress
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.enums import ComputeStep


def make_tracker() -> JobTracker:
    # The stale-job sweeper thread is not under test
    with patch("language_pipes.jobs.job_tracker.Thread"):
        return JobTracker()


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


class AddJobTests(unittest.TestCase):
    """`add_job` builds the local record for a job this node only hosts layers
    for, so anything the origin owns has to stay unset here."""

    def test_prompt_tokens_is_not_guessed_from_the_state_in_flight(self):
        tracker = make_tracker()

        job = tracker.add_job(make_network_job(chunk_width=8), PretrainedConfig(num_hidden_layers=1))

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

        job = tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1))
        assert job is not None
        job.receive_network_job(network_job, "node-b")

        self.assertEqual(job.display_progress().prompt_tokens, 200)
        self.assertEqual(job.display_progress().prefill_tokens, 64)

    def test_rejects_a_job_it_already_tracks(self):
        tracker = make_tracker()
        config = PretrainedConfig(num_hidden_layers=1)

        self.assertIsNotNone(tracker.add_job(make_network_job(), config))
        self.assertIsNone(tracker.add_job(make_network_job(), config))

    def test_rejects_a_job_that_was_never_embedded(self):
        tracker = make_tracker()
        network_job = make_network_job()
        network_job.data.state = None  # pyright: ignore[reportOptionalMemberAccess]

        with self.assertRaises(Exception):
            tracker.add_job(network_job, PretrainedConfig(num_hidden_layers=1))


if __name__ == "__main__":
    unittest.main()
