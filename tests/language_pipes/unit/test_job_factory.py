import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from transformers import PretrainedConfig

from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker


class FakeCollector:
    config = PretrainedConfig(num_hidden_layers=1)


class FakeEndModel:
    def __init__(self):
        self.layers = []
        self.collector = FakeCollector()


class FakePipe:
    pipe_id = "pipe-1"
    model_id = "model-1"

    def send_job(self, network_job, node_id):
        pass


class FakeModelManager:
    def get_end_model(self, model_id):
        return FakeEndModel()


class FakeRouter:
    def node_id(self):
        return "node-a"


class FakeRouterPipes:
    router = FakeRouter()


class FakePipeManager:
    def __init__(self):
        self.model_manager = FakeModelManager()
        self.router_pipes = FakeRouterPipes()

    def get_pipe_by_model_id(self, model_id, start_layer=0):
        return FakePipe()


def make_factory(max_api_jobs: int = 5):
    tracker = JobTracker()
    tracker.shutdown = True  # stop the stale-job background thread
    return JobFactory(tracker, FakePipeManager(), lambda: max_api_jobs)  # pyright: ignore[reportArgumentType]


class MaxApiJobsTests(unittest.TestCase):
    def test_rejects_when_key_over_limit(self):
        factory = make_factory(max_api_jobs=2)
        # Pre-fill the key past the limit (limit is 2, guard trips at > 2).
        factory.job_tracker.jobs_pending["key-1"] = ["j0", "j1", "j2"] # pyright: ignore[reportArgumentType]

        resolved = []
        factory.start_job(
            "key-1", "model-1", [], max_completion_tokens=8,
            resolve=lambda v: resolved.append(v), # pyright: ignore[reportArgumentType]
        )

        self.assertEqual(resolved, ["MAX_JOBS"])

    def test_allows_when_under_limit(self):
        factory = make_factory(max_api_jobs=2)

        resolved = []
        job = factory.start_job(
            "key-1", "model-1", [], max_completion_tokens=8,
            resolve=lambda v: resolved.append(v), # pyright: ignore[reportArgumentType]
        )

        self.assertIsNotNone(job)
        self.assertNotIn("MAX_JOBS", resolved)
        self.assertEqual(len(factory.job_tracker.jobs_pending["key-1"]), 1)

    def test_limit_is_per_api_key(self):
        factory = make_factory(max_api_jobs=2)
        factory.job_tracker.jobs_pending["key-1"] = ["j0", "j1", "j2"] # pyright: ignore[reportArgumentType]

        resolved = []
        # A different key is unaffected by key-1 being over the limit.
        job = factory.start_job(
            "key-2", "model-1", [], max_completion_tokens=8,
            resolve=lambda v: resolved.append(v), # pyright: ignore[reportArgumentType]
        )

        self.assertIsNotNone(job)
        self.assertEqual(len(factory.job_tracker.jobs_pending["key-2"]), 1)


if __name__ == "__main__":
    unittest.main()
