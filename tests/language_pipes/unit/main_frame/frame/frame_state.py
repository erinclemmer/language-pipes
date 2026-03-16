import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.frame.frame_state import FrameState

class TestFrameState(unittest.TestCase):
    """Unit tests for FrameState in isolation."""

    def test_initial_state(self):
        state = FrameState()
        self.assertFalse(state.running)
        self.assertFalse(state.exit_tui)
        self.assertEqual(state.status_message, "")
        self.assertEqual(state.status_level, "info")

    def test_set_status(self):
        state = FrameState()
        state.set_status("hello", "warning")
        self.assertEqual(state.status_message, "hello")
        self.assertEqual(state.status_level, "warning")

    def test_clear_status(self):
        state = FrameState()
        state.set_status("something", "error")
        state.clear_status()
        self.assertEqual(state.status_message, "")
        self.assertEqual(state.status_level, "info")

    def test_startup(self):
        state = FrameState()
        state.startup()
        self.assertTrue(state.running)
        self.assertFalse(state.exit_tui)
