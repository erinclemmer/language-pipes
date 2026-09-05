import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'language_pipes', 'unit'))

import torch

from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.enums import ComputeStep

from util import (
    make_processor,
    make_job,
    mock_complete,
    FakeEndModelContinue,
    FakeModel,
    PipeWrapper,
)

ORIGIN = "node-1"
RELAY = "node-b"


def append_position(job):
    """Add one position to the node's KV cache, the way a real layer does."""
    keys = torch.zeros((1, 1, 1, 1))
    job.cache.update(keys, keys.clone(), 0)


class CachingEndModel(FakeEndModelContinue):
    """The origin's own layer segment. It never stops on the EOS token, so the
    job keeps decoding."""

    def compute_layers(self, job):
        append_position(job)
        super().compute_layers(job)


class CachingModel(FakeModel):
    """A layer segment on a node that hosts nothing else."""

    def process_job(self, job):
        append_position(job)
        super().process_job(job)


def transport(network_job: NetworkJob) -> NetworkJob:
    """Put the packet through the wire format, the way the router does."""
    restored, valid = NetworkJob.from_bytes(network_job.to_bytes())
    assert valid
    return restored


def corrupt(network_job: NetworkJob) -> NetworkJob:
    """A packet that failed its hash, as `JobReceiver.restart_token` returns it."""
    bounced = transport(network_job)
    bounced.data = None
    bounced.data_hash = b''
    bounced.compute_step = ComputeStep.EMBED
    bounced.current_layer = 0
    return bounced


class TwoNodePipe:
    """An origin holding the end model and layer 0, and one node holding layer 1."""

    def __init__(self):
        self.origin = make_job(complete=mock_complete)
        self.origin.origin_node_id = ORIGIN
        self.origin.compute_step = ComputeStep.TOKENIZE
        self.origin_pipe = PipeWrapper(
            ORIGIN, "model-a", [FakeModel(RELAY, 1, 1, virtual=True, num_hidden_layers=2)]
        )
        self.origin_end_model = CachingEndModel(num_local_layers=1)

        self.relay = make_job()
        self.relay.job_id = self.origin.job_id
        self.relay.pipe_id = self.origin.pipe_id
        self.relay.origin_node_id = ORIGIN
        self.relay_model = CachingModel(RELAY, 1, 1, virtual=False, num_hidden_layers=2)
        self.relay_pipe = PipeWrapper(RELAY, "model-a", [self.relay_model])

    def run_origin(self):
        make_processor(
            job=self.origin,
            pipe=self.origin_pipe,
            end_model=self.origin_end_model,
            node_id=ORIGIN
        ).run()
        return self.origin_pipe.sent_jobs[-1]

    def run_relay(self, packet: NetworkJob):
        self.assertion_hook = self.relay.receive_network_job(transport(packet), RELAY)
        assert self.assertion_hook
        make_processor(
            job=self.relay, pipe=self.relay_pipe, end_model=None, node_id=RELAY
        ).run()
        return self.relay_pipe.sent_jobs[-1]

    def deliver_to_origin(self, packet: NetworkJob) -> bool:
        return self.origin.receive_network_job(transport(packet), ORIGIN)

    def full_pass(self):
        """One pass all the way round: origin -> relay -> origin."""
        self.deliver_to_origin(self.run_relay(self.run_origin()))

    def positions(self):
        return (self.origin.cache.get_seq_length(), self.relay.cache.get_seq_length())


class TestRestartKeepsTheCachesInStep(unittest.TestCase):
    """A packet that fails its hash is bounced to the origin. Every node's cache
    must end up holding each position exactly once, whichever side of the
    corruption point it sits on."""

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_a_chunk_lost_on_the_way_out_is_computed_once(self):
        # 2 prompt tokens at a chunk size of 1 => two prefill chunks
        pipe = TwoNodePipe()
        sent = pipe.run_origin()

        # The relay cannot validate the packet and sends it back stripped
        self.assertTrue(pipe.deliver_to_origin(corrupt(sent)))
        resent = pipe.run_origin()

        self.assertEqual(resent.pass_idx, sent.pass_idx)
        self.assertEqual(pipe.origin.chunking.current_chunk, 0)

        pipe.deliver_to_origin(pipe.run_relay(resent))

        self.assertEqual(pipe.positions(), (1, 1))

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_a_chunk_lost_on_the_way_back_is_not_computed_twice(self):
        # The relay already ran its layers for this pass, so on the resend it
        # must forward what it produced rather than run them again.
        pipe = TwoNodePipe()
        returned = pipe.run_relay(pipe.run_origin())
        self.assertEqual(pipe.positions(), (1, 1))

        self.assertTrue(pipe.deliver_to_origin(corrupt(returned)))
        pipe.deliver_to_origin(pipe.run_relay(pipe.run_origin()))

        self.assertEqual(pipe.positions(), (1, 1))
        self.assertEqual(pipe.origin.chunking.current_chunk, 0)

    @patch("language_pipes.util.chunk_state.CHUNK_SIZE", 1)
    def test_the_second_chunk_still_runs_after_the_first_was_restarted(self):
        pipe = TwoNodePipe()
        pipe.full_pass()
        self.assertEqual(pipe.positions(), (1, 1))

        # The second chunk goes out and is bounced back
        pipe.deliver_to_origin(corrupt(pipe.run_origin()))
        pipe.deliver_to_origin(pipe.run_relay(pipe.run_origin()))

        # Two chunks, one position each, neither skipped nor run twice
        self.assertEqual(pipe.positions(), (2, 2))
        self.assertEqual(pipe.origin.chunking.current_chunk, 1)
        self.assertEqual(pipe.origin.prompt_tokens, 2)

    def test_a_bounced_decode_token_lands_in_every_cache_once(self):
        # A short prompt fits one chunk, so the second pass is a decode token
        pipe = TwoNodePipe()
        pipe.full_pass()
        self.assertEqual(pipe.positions(), (1, 1))

        # The head runs, the first decode token is embedded and sent out
        returned = pipe.run_relay(pipe.run_origin())
        self.assertEqual(pipe.origin.current_token, 1)
        self.assertEqual(pipe.positions(), (2, 2))

        self.assertTrue(pipe.deliver_to_origin(corrupt(returned)))
        pipe.deliver_to_origin(pipe.run_relay(pipe.run_origin()))

        self.assertEqual(pipe.positions(), (2, 2))
        self.assertEqual(pipe.origin.current_token, 1)
        self.assertEqual(len(pipe.origin.input_ids), 3)


if __name__ == "__main__":
    unittest.main()
