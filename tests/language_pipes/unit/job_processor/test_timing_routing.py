import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.util.enums import ComputeStep

from language_pipes.jobs.network_job import NetworkJob

from util import make_processor, make_job, mock_complete, FakeEndModel, FakeEndModelContinue, FakeModel, PipeWrapper

def run_job(end_model, **job_kwargs):
    job = make_job(complete=mock_complete, **job_kwargs)
    job.origin_node_id = "node-1"
    job.compute_step = ComputeStep.TOKENIZE

    model = FakeModel("node-a", 0, 1, virtual=False, num_hidden_layers=2)
    pipe = PipeWrapper("node-1", "model-a", [model])
    processor = make_processor(job=job, pipe=pipe, end_model=end_model)
    processor.run()
    return job

class TestPrefillTimingRouting(unittest.TestCase):
    """Prefill passes must land in prefill_times, never in the decode averages."""

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_final_chunk_does_not_pollute_decode_times(self):
        # 2 prompt tokens at a chunk size of 1 => two prefill chunks
        job = run_job(FakeEndModel())

        self.assertEqual(len(job.timing_stats.prefill_times.token_ms), 2)
        self.assertEqual(job.timing_stats.prefill_times.token_counts, [1, 1])
        self.assertEqual(job.timing_stats.output_times.token_ms, [])

    def test_unchunked_prefill_recorded_with_full_prompt_length(self):
        # Prompt fits in a single chunk, so chunking never activates
        job = run_job(FakeEndModel())

        self.assertFalse(job.chunking.is_active())
        self.assertEqual(job.timing_stats.prefill_times.token_counts, [job.prompt_tokens])
        self.assertEqual(job.timing_stats.output_times.token_ms, [])

    def test_decode_tokens_recorded_after_prefill(self):
        job = run_job(FakeEndModelContinue(), max_completion_tokens=3)

        # One prefill pass, then the remaining passes are decode tokens
        self.assertEqual(len(job.timing_stats.prefill_times.token_ms), 1)
        self.assertEqual(len(job.timing_stats.output_times.token_ms), 2)
        self.assertEqual(job.timing_stats.output_times.token_counts, [1, 1])

    def test_decode_stats_are_populated_for_short_prompts(self):
        job = run_job(FakeEndModelContinue(), max_completion_tokens=2)

        # The regression: these read 0.0 when sourced from prefill_times
        self.assertGreater(job.timing_stats.output_times.get_avg_embed_time(), 0)
        self.assertGreater(job.timing_stats.output_times.get_avg_layer_time(), 0)
        self.assertGreater(job.timing_stats.output_times.get_tokens_per_second(), 0)

class TestCompletedPassHandoff(unittest.TestCase):
    """Nodes past the origin only learn a pass finished when the origin tells
    them, so every job the origin sends out carries the pass it just closed."""

    def make_origin(self, end_model):
        job = make_job(complete=mock_complete)
        job.origin_node_id = "node-1"
        job.compute_step = ComputeStep.TOKENIZE

        # The layers live on another node, so the origin has to send the job out
        remote = FakeModel("node-b", 0, 1, virtual=True, num_hidden_layers=2)
        pipe = PipeWrapper("node-1", "model-a", [remote])
        return job, pipe, make_processor(job=job, pipe=pipe, end_model=end_model)

    def return_from_pipe(self, job):
        """Hand the job back to the origin the way the last node on the pipe would."""
        job.receive_network_job(NetworkJob(
            job_id=job.job_id,
            pipe_id=job.pipe_id,
            origin_node_id=job.origin_node_id,
            current_layer=0,
            data=job.data,
            data_hash=b'',
            compute_step=ComputeStep.HEAD,
            times=list(job.timing_stats.current_times),
            completed=job.timing_stats.completed_pass,
            progress=job.get_progress()
        ), job.origin_node_id)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_first_send_has_no_pass_yet(self):
        job, pipe, processor = self.make_origin(FakeEndModel())
        processor.run()

        self.assertEqual(len(pipe.sent_jobs), 1)
        self.assertIsNone(pipe.sent_jobs[0].completed)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_next_send_carries_the_finished_prefill_chunk(self):
        # 2 prompt tokens at a chunk size of 1 => the first chunk finishes before
        # the second one is embedded and sent back out
        job, pipe, processor = self.make_origin(FakeEndModel())
        processor.run()

        self.return_from_pipe(job)
        make_processor(job=job, pipe=pipe, end_model=FakeEndModel()).run()

        self.assertEqual(len(pipe.sent_jobs), 2)
        completed = pipe.sent_jobs[1].completed
        self.assertIsNotNone(completed)
        self.assertTrue(completed.is_prefill)
        self.assertEqual(completed.token_count, 1)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_relay_reports_the_same_speed_as_the_origin(self):
        job, pipe, processor = self.make_origin(FakeEndModel())
        processor.run()

        self.return_from_pipe(job)
        make_processor(job=job, pipe=pipe, end_model=FakeEndModel()).run()

        relay = make_job()
        relay.job_id = job.job_id
        relay.pipe_id = job.pipe_id
        relay.origin_node_id = job.origin_node_id
        restored, _ = NetworkJob.from_bytes(pipe.sent_jobs[1].to_bytes())
        relay.receive_network_job(restored, "node-b")

        self.assertGreater(relay.timing_stats.prefill_times.get_tokens_per_second(), 0)
        self.assertEqual(
            relay.timing_stats.prefill_times.get_tokens_per_second(),
            job.timing_stats.prefill_times.get_tokens_per_second()
        )

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_relay_reports_the_same_prefill_position_as_the_origin(self):
        job, pipe, processor = self.make_origin(FakeEndModel())
        processor.run()

        # Second chunk in flight, so the first one counts as processed
        self.return_from_pipe(job)
        make_processor(job=job, pipe=pipe, end_model=FakeEndModel()).run()

        relay = make_job()
        relay.job_id = job.job_id
        relay.pipe_id = job.pipe_id
        relay.origin_node_id = job.origin_node_id
        restored, _ = NetworkJob.from_bytes(pipe.sent_jobs[1].to_bytes())
        relay.receive_network_job(restored, "node-b")

        origin_progress = job.display_progress()
        relay_progress = relay.display_progress()
        self.assertTrue(relay_progress.prefilling)
        self.assertEqual(relay_progress.prefill_tokens, 1)
        self.assertEqual(relay_progress.prefill_tokens, origin_progress.prefill_tokens)
        self.assertEqual(relay_progress.prompt_tokens, origin_progress.prompt_tokens)

if __name__ == "__main__":
    unittest.main()
