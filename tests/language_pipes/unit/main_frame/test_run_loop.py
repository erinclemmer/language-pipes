import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.frame_state import FrameState
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _noop_print_pos, _noop_write, _PRINT_POS_PATCH, _READ_KEY_PATCH, _WRITE_PATCH

class TestRunLoop(unittest.TestCase):
    """Verify that run() starts the state machine and returns correct codes."""

    def test_run_returns_exit_when_exit_tui(self):
        """run() should return 'exit' when state.exit_tui is True."""
        frame = _make_main_frame()

        # Simulate: first key opens exit confirm, second selects "Exit TUI",
        # third confirms.
        call_count = [0]
        def fake_read_key():
            call_count[0] += 1
            if call_count[0] == 1:
                return PressedKey.Alpha, "q"       # open exit confirm
            elif call_count[0] == 2:
                return PressedKey.ArrowUp, ""       # move to "Exit TUI"
            elif call_count[0] == 3:
                return PressedKey.Enter, ""         # confirm
            return PressedKey.Nop, ""

        with patch(_WRITE_PATCH, _noop_write), \
             patch(_PRINT_POS_PATCH, _noop_print_pos), \
             patch(_READ_KEY_PATCH, fake_read_key):
            result = frame.run()

        self.assertEqual(result, "exit")

    def test_run_returns_menu_when_return_to_menu(self):
        """run() should return 'menu' when 'Return to menu' is chosen."""
        frame = _make_main_frame()

        call_count = [0]
        def fake_read_key():
            call_count[0] += 1
            if call_count[0] == 1:
                return PressedKey.Alpha, "q"       # open exit confirm
            elif call_count[0] == 2:
                # ExitConfirm defaults to Cancel (idx=2).
                # "Return to menu" is idx=0. Move up twice.
                return PressedKey.ArrowUp, ""
            elif call_count[0] == 3:
                return PressedKey.ArrowUp, ""
            elif call_count[0] == 4:
                return PressedKey.Enter, ""
            return PressedKey.Nop, ""

        with patch(_WRITE_PATCH, _noop_write), \
             patch(_PRINT_POS_PATCH, _noop_print_pos), \
             patch(_READ_KEY_PATCH, fake_read_key):
            result = frame.run()

        self.assertEqual(result, "menu")

    def test_run_cancel_exit_keeps_running(self):
        """Cancelling exit confirm should keep the loop running."""
        frame = _make_main_frame()

        call_count = [0]
        def fake_read_key():
            call_count[0] += 1
            if call_count[0] == 1:
                return PressedKey.Alpha, "q"       # open exit confirm
            elif call_count[0] == 2:
                return PressedKey.Escape, ""        # cancel -> keep running
            elif call_count[0] == 3:
                # Now actually exit
                return PressedKey.Alpha, "q"
            elif call_count[0] == 4:
                return PressedKey.ArrowUp, ""       # -> Exit TUI
            elif call_count[0] == 5:
                return PressedKey.Enter, ""
            return PressedKey.Nop, ""

        with patch(_WRITE_PATCH, _noop_write), \
             patch(_PRINT_POS_PATCH, _noop_print_pos), \
             patch(_READ_KEY_PATCH, fake_read_key):
            result = frame.run()

        self.assertEqual(result, "exit")
        # Verify we went through at least 5 key reads (cancel + re-exit)
        self.assertGreaterEqual(call_count[0], 5)

    def test_startup_sets_running(self):
        """FrameState.startup() should set running=True, exit_tui=False."""
        state = FrameState()
        self.assertFalse(state.running)
        state.startup()
        self.assertTrue(state.running)
        self.assertFalse(state.exit_tui)
