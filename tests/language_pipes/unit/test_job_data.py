import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.jobs.job_data import JobData


class JobDataTests(unittest.TestCase):
    def test_hash_and_validate_state(self):
        job_data = JobData(
            state=torch.zeros((1, 4)),
            position_ids=torch.tensor([0, 1, 2, 3]),
            cache_position=torch.tensor([]),
            causal_mask=torch.tensor([]),
            position_embeddings=None,
            position_embeddings_global=None,
            position_embeddings_local=None,
            sliding_causal_mask=None
        )
        state_hash = job_data.hash_state()

        self.assertTrue(JobData.validate_state(job_data.to_bytes(), state_hash))
        self.assertFalse(job_data.validate_state(job_data.to_bytes(), b"bad-hash"))

    def test_round_trip_bytes(self):
        job_data = JobData(
            state=torch.ones((2, 2)),
            position_ids=torch.tensor([0, 1]),
            cache_position=torch.tensor([1]),
            causal_mask=torch.tensor([]),
            position_embeddings=None,
            position_embeddings_global=None,
            position_embeddings_local=None,
            sliding_causal_mask=None
        )

        serialized = job_data.to_bytes()
        restored = JobData.from_bytes(serialized)
        self.assertIsNotNone(restored)
        if restored is not None:
            self.assertTrue(torch.equal(restored.state, job_data.state))
            self.assertTrue(torch.equal(restored.position_ids, job_data.position_ids))
            self.assertTrue(torch.equal(restored.cache_position, job_data.cache_position))


if __name__ == "__main__":
    unittest.main()
