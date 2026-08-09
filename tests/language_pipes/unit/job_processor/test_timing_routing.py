import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.util.enums import ComputeStep

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

if __name__ == "__main__":
    unittest.main()
