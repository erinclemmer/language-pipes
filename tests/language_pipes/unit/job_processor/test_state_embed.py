import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.jobs.job_processor import JobState
from language_pipes.util.enums import ComputeStep

from util import make_processor, make_job, make_config, FakeEndModel, FakeModel, PipeWrapper

class TestEmbedState(unittest.TestCase):
    """Tests for the _state_embed method."""

    def test_transitions_to_done_when_update_fails(self):
        updates = []

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update)
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 2
        job.current_token = 0
        job.init_chunking(chunk_size=1)

        processor = make_processor(
            job=job,
            config=make_config(prefill_chunk_size=1),
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)
        self.assertEqual(len(updates), 1)

    def test_transitions_to_done_when_model_missing(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.DONE)

    def test_transitions_to_send_for_remote_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        virtual_model = FakeModel("node-b", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(job=job, pipe=pipe, end_model=FakeEndModel())

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.SEND)

    def test_transitions_to_process_layers_for_local_layer(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        job.prompt_tokens = 1

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_process_layers_for_prefill(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED

        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])
        processor = make_processor(
            job=job,
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=0),
        )

        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

        job.compute_step = ComputeStep.EMBED
        next_state = processor._state_embed()

        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_process_layers_for_num_local_layers(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED

        virtual_model = FakeModel("node-a", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(
            job=job,
            # Simulate node loading the same layer as the EndModel
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=1)
        )

        next_state = processor._state_embed()

        # The job should not be sent to the virtual node
        self.assertEqual(next_state, JobState.PROCESS_LAYERS)

    def test_transitions_to_send_without_local_layers(self):
        job = make_job()
        job.compute_step = ComputeStep.EMBED
        virtual_model = FakeModel("node-a", 0, 0, virtual=True, num_hidden_layers=2)
        local_model = FakeModel("node-a", 1, 1, virtual=False, num_hidden_layers=2)
        pipe = PipeWrapper("node-a", "model-a", [virtual_model, local_model])
        processor = make_processor(
            job=job,
            # Simulate node loading the same layer as the EndModel
            pipe=pipe,
            end_model=FakeEndModel(num_local_layers=0)
        )

        next_state = processor._state_embed()

        # The job should be sent to the virtual node if no local layers are specified
        self.assertEqual(next_state, JobState.SEND)

class TestEmbedPrefillIntegration(unittest.TestCase):
    """Integration tests for embed state during prefill operations."""

    def test_update_failure_stops(self):
        updates = []
        
        def complete(_):
            pass

        def fail_update(job):
            updates.append(job.compute_step)
            return False

        job = make_job(update=fail_update, complete=complete)
        job.compute_step = ComputeStep.TOKENIZE
        end_model = FakeEndModel()
        model = FakeModel("node-a", 0, 0, virtual=False, num_hidden_layers=1)
        pipe = PipeWrapper("node-a", "model-a", [model])

        processor = make_processor(
            job=job,
            pipe=pipe,
            config=make_config(prefill_chunk_size=1),
            end_model=end_model,
        )

        processor.run()

        self.assertEqual(
            processor.states,
            [JobState.VALIDATING, JobState.EMBED, JobState.PROCESS_LAYERS, JobState.EMBED],
        )
        self.assertEqual(processor.state, JobState.DONE)
        self.assertIn("tokenize", end_model.calls)
        self.assertIn("compute_embed", end_model.calls)

if __name__ == "__main__":
    unittest.main()