import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch

from language_pipes.content_provider.model_provider import ModelStatusInfo, ModelStatus
from language_pipes.tui.components.hosted_models_view import format_pipe_strings


def make_running(ram_bytes: float):
    return [
        ModelStatusInfo(
            status=ModelStatus.Running,
            device=torch.device("cpu"),
            pipe_id="pipe-1",
            start_layer=0,
            end_layer=4,
            num_layers=5,
            end_model=False,
            ram_used=ram_bytes,
        )
    ]


class EightBitRamTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_reports_full_ram_when_not_8bit(self):
        lines = format_pipe_strings(make_running(2 * 1024 ** 3))
        self.assertEqual(len(lines), 1)
        self.assertIn("(2.00GB)", lines[0])

    @mock.patch.dict(os.environ, {"LP_8_BIT_MODE": "true"}, clear=True)
    def test_halves_reported_ram_in_8bit_mode(self):
        lines = format_pipe_strings(make_running(2 * 1024 ** 3))
        self.assertEqual(len(lines), 1)
        self.assertIn("(1.00GB)", lines[0])


if __name__ == "__main__":
    unittest.main()
