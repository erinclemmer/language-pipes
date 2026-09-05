import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from util import make_processor, make_job, make_job_data, FakeModel, PipeWrapper

class TestSendState(unittest.TestCase):
    """Tests for the _state_send method."""

    def test_transitions_to_done_after_handoff(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 0
        job.data = make_job_data()

        virtual_model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(job=job, pipe=pipe, end_model=None)

        next_state = processor._state_send()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(pipe.calls, ["send job"])

    def test_routes_tokenize_job_to_next_node_when_origin_mismatch(self):
        job = make_job(origin_node_id="node-b")
        job.compute_step = ComputeStep.TOKENIZE
        job.current_layer = 0
        job.data = make_job_data()

        next_model = FakeModel("node-c", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [next_model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=None,
        )

        processor.run()

        self.assertEqual(processor.states, [JobState.VALIDATING, JobState.SEND])
        self.assertEqual(processor.state, JobState.DONE)
        self.assertEqual(pipe.calls, ["send job"])


class TestSendSavesThePass(unittest.TestCase):
    """Whatever goes out is kept, so that a node that cannot validate it can
    have it sent again without anything being computed twice."""

    def make_job_at_send(self):
        job = make_job()
        job.compute_step = ComputeStep.LAYER
        job.current_layer = 1
        job.data = make_job_data()
        job.pass_idx = 4
        job.pass_key = (ComputeStep.LAYER, 0)

        next_model = FakeModel("node-b", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [next_model])
        return job, pipe

    def test_the_pass_number_goes_on_the_wire(self):
        job, pipe = self.make_job_at_send()

        make_processor(job=job, pipe=pipe, end_model=None)._state_send()

        self.assertEqual(pipe.sent_jobs[0].pass_idx, 4)

    def test_what_went_out_is_saved_under_the_entry_point(self):
        job, pipe = self.make_job_at_send()

        make_processor(job=job, pipe=pipe, end_model=None)._state_send()

        saved = job.pass_outputs[(ComputeStep.LAYER, 0)]
        self.assertEqual(saved.pass_idx, 4)
        self.assertIs(saved.data, job.data)
        self.assertEqual(saved.compute_step, ComputeStep.LAYER)
        self.assertEqual(saved.current_layer, 1)

    def test_a_replay_ends_when_the_pass_goes_back_out(self):
        job, pipe = self.make_job_at_send()
        job.replaying = True

        make_processor(job=job, pipe=pipe, end_model=None)._state_send()

        self.assertFalse(job.replaying)


if __name__ == "__main__":
    unittest.main()
