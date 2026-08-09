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
