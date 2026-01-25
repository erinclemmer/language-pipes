import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.jobs.job_data import JobData


class JobDataTests(unittest.TestCase):
    def test_hash_and_validate_state(self):
        job_data = JobData()
        job_data.state = torch.zeros((1, 4))
        job_data.position_ids = torch.tensor([[0, 1, 2, 3]])

        state_hash = job_data.hash_state()

        self.assertTrue(JobData.validate_state(job_data.to_bytes(), state_hash))
        self.assertFalse(job_data.validate_state(job_data.to_bytes(), b"bad-hash"))

    def test_round_trip_bytes(self):
        job_data = JobData()
        job_data.state = torch.ones((2, 2))
        job_data.position_ids = torch.tensor([[0, 1]])
        job_data.cache_position = torch.tensor([1])

        serialized = job_data.to_bytes()
        restored = JobData.from_bytes(serialized)

        self.assertTrue(torch.equal(restored.state, job_data.state))
        self.assertTrue(torch.equal(restored.position_ids, job_data.position_ids))
        self.assertTrue(torch.equal(restored.cache_position, job_data.cache_position))


if __name__ == "__main__":
    unittest.main()
