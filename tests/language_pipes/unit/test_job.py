import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.jobs.job import Job
from language_pipes.util.enums import ComputeStep, JobStatus


def make_job():
    return Job(
        origin_node_id="node-a",
        messages=[],
        pipe_id="pipe-1",
        model_id="model-1",
        prefill_chunk_size=6,
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


if __name__ == "__main__":
    unittest.main()
