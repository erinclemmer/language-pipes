import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from util import make_processor, make_job, make_job_data, FakeEndModel, FakeModel, PipeWrapper


class FailRecorder:
    """Stands in for JobReceiver.cancel_job."""

    def __init__(self):
        self.calls = []

    def __call__(self, job, reason):
        self.calls.append((job.job_id, reason))
        job.cancel_reason = reason


class TestFailsWhenLayerHasNoHost(unittest.TestCase):
    """A model unloaded elsewhere leaves the pipe short a segment - the job has
    to be cancelled rather than dropped for the stale timeout to notice."""

    def test_validating_fails_when_no_node_hosts_the_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = make_job_data()

        on_fail = FailRecorder()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0, num_hidden_layers=1)])
        processor = make_processor(job=job, pipe=pipe, end_model=None, on_fail=on_fail)

        next_state = processor._state_validating()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(on_fail.calls, [(job.job_id, "no node hosts layer 1")])

    def test_process_layers_fails_when_no_node_hosts_the_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 2
        job.data = make_job_data()

        on_fail = FailRecorder()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0, num_hidden_layers=3)])
        processor = make_processor(job=job, pipe=pipe, end_model=None, on_fail=on_fail)

        next_state = processor._state_process_layers()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(on_fail.calls, [(job.job_id, "no node hosts layer 2")])

    def test_send_fails_when_next_segment_is_gone(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = make_job_data()

        on_fail = FailRecorder()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0, num_hidden_layers=2)])
        processor = make_processor(job=job, pipe=pipe, end_model=None, on_fail=on_fail)

        next_state = processor._state_send()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(on_fail.calls, [(job.job_id, "no node hosts layer 1")])
        self.assertEqual(pipe.calls, [])

    def test_head_fails_when_end_model_was_unloaded(self):
        job = make_job()
        job.compute_step = ComputeStep.HEAD
        job.data = make_job_data()

        on_fail = FailRecorder()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0)])
        processor = make_processor(job=job, pipe=pipe, end_model=None, on_fail=on_fail)

        next_state = processor._state_head()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(on_fail.calls, [(job.job_id, "end model unloaded")])

    def test_embed_fails_when_end_model_was_unloaded(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.data = make_job_data()

        on_fail = FailRecorder()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0)])
        processor = make_processor(job=job, pipe=pipe, end_model=None, on_fail=on_fail)

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(on_fail.calls, [(job.job_id, "end model unloaded")])

    def test_missing_layer_without_a_handler_still_stops(self):
        # on_fail is optional; the job must not loop when nothing is wired up.
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = make_job_data()

        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0, num_hidden_layers=1)])
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        processor.run()

        self.assertEqual(processor.state, JobState.DONE)


class TestStopsOnceCanceled(unittest.TestCase):
    def test_run_stops_when_the_job_was_canceled(self):
        job = make_job(origin_node_id="node-1")
        job.compute_step = ComputeStep.TOKENIZE
        job.cancel_reason = "layers for model-a unloaded"

        end_model = FakeEndModel()
        pipe = PipeWrapper("node-a", "model-a", [FakeModel("node-a", 0, 0)])
        processor = make_processor(job=job, pipe=pipe, end_model=end_model)

        processor.run()

        self.assertEqual(processor.states, [])
        self.assertEqual(end_model.calls, [])


if __name__ == "__main__":
    unittest.main()
