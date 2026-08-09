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


class FailingPipe(FakePipe):
    def send_job(self, network_job, node_id):
        raise Exception("node unreachable")


class RecordingPipe(FakePipe):
    """Captures what the tracker knew at the moment the job was dispatched."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.tracked_on_send = None

    def send_job(self, network_job, node_id):
        self.tracked_on_send = self.tracker.get_job(network_job.job_id)


class FakeModelManager:
    def get_end_model(self, model_id):
        return FakeEndModel()


class FakeRouter:
    def node_id(self):
        return "node-a"


class FakeRouterPipes:
    router = FakeRouter()


class FakePipeManager:
    def __init__(self, pipe=None):
        self.model_manager = FakeModelManager()
        self.router_pipes = FakeRouterPipes()
        self.pipe = pipe or FakePipe()

    def get_pipe_by_model_id(self, model_id, start_layer=0):
        return self.pipe


def make_factory(max_api_jobs: int = 5, pipe=None):
    tracker = JobTracker()
    tracker.shutdown = True  # stop the stale-job background thread
    return JobFactory(tracker, FakePipeManager(pipe), lambda: max_api_jobs)  # pyright: ignore[reportArgumentType]


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


class DispatchOrderTests(unittest.TestCase):
    def test_job_is_tracked_before_it_is_sent(self):
        # The first hop can be this same node, so a job that is cancelled or
        # completed during send must already be findable in the tracker.
        tracker = JobTracker()
        tracker.shutdown = True
        pipe = RecordingPipe(tracker)
        factory = JobFactory(tracker, FakePipeManager(pipe), lambda: 5)  # pyright: ignore[reportArgumentType]

        job = factory.start_job("key-1", "model-1", [], max_completion_tokens=8)

        self.assertIsNotNone(job)
        self.assertIs(pipe.tracked_on_send, job)

    def test_start_callback_runs_for_a_dispatched_job(self):
        factory = make_factory()
        started = []

        factory.start_job(
            "key-1", "model-1", [], max_completion_tokens=8,
            start=lambda j: started.append(j),
        )

        self.assertEqual(len(started), 1)

    def test_failed_dispatch_cancels_instead_of_leaving_the_caller_waiting(self):
        factory = make_factory(pipe=FailingPipe())
        resolved = []

        job = factory.start_job(
            "key-1", "model-1", [], max_completion_tokens=8,
            resolve=lambda v: resolved.append(v), # pyright: ignore[reportArgumentType]
        )

        self.assertIsNone(job)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].cancel_reason, "could not send job to pipe")
        self.assertEqual(factory.job_tracker.jobs_pending["key-1"], [])


if __name__ == "__main__":
    unittest.main()
