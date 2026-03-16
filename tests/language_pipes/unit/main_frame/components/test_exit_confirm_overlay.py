import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey

from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys


class TestExitConfirmOverlay(unittest.TestCase):
    """Verify exit confirmation dialog behaviour."""

    def setUp(self):
        self.frame = _make_main_frame()

    def test_exit_confirm_opens_on_q(self):
        _simulate_keys(self.frame, [(PressedKey.Alpha, "q")])
        self.assertTrue(self.frame.exit_confirm.is_open)

    def test_exit_confirm_cancel_closes(self):
        _simulate_keys(self.frame, [
            (PressedKey.Alpha, "q"),
            (PressedKey.Escape, ""),
        ])
        self.assertFalse(self.frame.exit_confirm.is_open)

    def test_exit_confirm_default_is_cancel(self):
        """Default selection in ExitConfirm should be 'Cancel'."""
        _simulate_keys(self.frame, [(PressedKey.Alpha, "q")])
        self.assertEqual(self.frame.exit_confirm.selected_option(), "Cancel")

    def test_exit_confirm_arrow_navigates(self):
        """Arrow keys should cycle through exit confirm options."""
        _simulate_keys(self.frame, [(PressedKey.Alpha, "q")])
        _simulate_keys(self.frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(self.frame.exit_confirm.selected_option(), "Exit TUI")

    def test_keys_ignored_while_exit_confirm_open(self):
        """Regular navigation keys should not affect nav state while exit confirm is open."""
        _simulate_keys(self.frame, [(PressedKey.Alpha, "q")])
        tab_before = self.frame.nav.active_tab()
        depth_before = self.frame.nav.focus_depth
        _simulate_keys(self.frame, [(PressedKey.ArrowRight, "")])
        self.assertEqual(self.frame.nav.active_tab(), tab_before)
        self.assertEqual(self.frame.nav.focus_depth, depth_before)
