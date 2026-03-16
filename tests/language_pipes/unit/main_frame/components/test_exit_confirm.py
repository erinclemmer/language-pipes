import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.exit_confirm import ExitConfirm

class TestExitConfirm(unittest.TestCase):
    """Unit tests for ExitConfirm in isolation."""

    def test_initial_state(self):
        ec = ExitConfirm()
        self.assertFalse(ec.is_open)

    def test_open_sets_default_to_cancel(self):
        ec = ExitConfirm()
        ec.open()
        self.assertTrue(ec.is_open)
        self.assertEqual(ec.selected_option(), "Cancel")

    def test_move_prev_wraps(self):
        ec = ExitConfirm()
        ec.open()
        ec.move_prev()  # Cancel -> Exit TUI
        self.assertEqual(ec.selected_option(), "Exit TUI")
        ec.move_prev()  # Exit TUI -> Return to menu
        self.assertEqual(ec.selected_option(), "Return to menu")
        ec.move_prev()  # Return to menu -> Cancel (wrap)
        self.assertEqual(ec.selected_option(), "Cancel")

    def test_handle_key_enter_returns_confirm(self):
        ec = ExitConfirm()
        ec.open()
        result = ec.handle_key(PressedKey.Enter)
        self.assertEqual(result, "confirm")

    def test_handle_key_escape_returns_cancel(self):
        ec = ExitConfirm()
        ec.open()
        result = ec.handle_key(PressedKey.Escape)
        self.assertEqual(result, "cancel")

    def test_render_contains_options(self):
        ec = ExitConfirm()
        ec.open()
        rendered = ec.render()
        for opt in ExitConfirm.OPTIONS:
            self.assertIn(opt, rendered)
