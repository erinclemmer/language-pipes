import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.main_frame import MainFrame
from language_pipes.tui.kb_utils import PressedKey
from language_pipes.tui.tui import TuiGrid


class MainFrameDispatchTests(unittest.TestCase):
    def setUp(self):
        self.paint_patch = patch.object(TuiGrid, "paint", return_value=None)
        self.paint_patch.start()

        self.frame = MainFrame((80, 24), (0, 0))
        self.frame.running = True

    def tearDown(self):
        self.paint_patch.stop()

    def test_focus_depth_enter_and_escape_transitions(self):
        self.assertEqual(self.frame.focus_depth, 0)

        self.frame._handle_key(PressedKey.Enter, "\n")
        self.assertEqual(self.frame.focus_depth, 1)

        self.frame._handle_key(PressedKey.Enter, "\n")
        self.assertEqual(self.frame.focus_depth, 2)

        self.frame._handle_key(PressedKey.Escape, "\x1b")
        self.assertEqual(self.frame.focus_depth, 1)

        self.frame._handle_key(PressedKey.Escape, "\x1b")
        self.assertEqual(self.frame.focus_depth, 0)

    def test_root_escape_opens_confirmation(self):
        self.frame.focus_depth = 0

        self.frame._handle_key(PressedKey.Escape, "\x1b")

        self.assertTrue(self.frame.confirm_escape_open)
        self.assertTrue(self.frame.running)
        self.assertEqual(self.frame.status_level, "warning")

    def test_root_confirmation_return_to_menu(self):
        self.frame.focus_depth = 0
        self.frame._handle_key(PressedKey.Escape, "\x1b")

        self.frame._handle_key(PressedKey.ArrowUp, "")  # Cancel -> Exit TUI
        self.frame._handle_key(PressedKey.ArrowUp, "")  # Exit TUI -> Return to menu
        self.frame._handle_key(PressedKey.Enter, "\n")

        self.assertFalse(self.frame.confirm_escape_open)
        self.assertFalse(self.frame.running)
        self.assertFalse(self.frame.exit_tui)

    def test_root_confirmation_exit_tui(self):
        self.frame.focus_depth = 0
        self.frame._handle_key(PressedKey.Escape, "\x1b")

        self.frame._handle_key(PressedKey.ArrowUp, "")  # Cancel -> Exit TUI
        self.frame._handle_key(PressedKey.Enter, "\n")

        self.assertFalse(self.frame.confirm_escape_open)
        self.assertFalse(self.frame.running)
        self.assertTrue(self.frame.exit_tui)

    def test_root_confirmation_cancel_via_escape(self):
        self.frame.focus_depth = 0
        self.frame._handle_key(PressedKey.Escape, "\x1b")

        self.frame._handle_key(PressedKey.Escape, "\x1b")

        self.assertFalse(self.frame.confirm_escape_open)
        self.assertTrue(self.frame.running)
        self.assertEqual(self.frame.status_message, "Exit canceled")

    def test_refresh_sets_placeholder_status(self):
        self.frame._handle_key(PressedKey.Alpha, "r")
        self.assertEqual(self.frame.status_message, "Refreshed (placeholder view)")
        self.assertEqual(self.frame.status_level, "info")

    def test_side_selection_retained_per_top_tab(self):
        self.frame.focus_depth = 1

        self.frame._handle_key(PressedKey.ArrowDown, "")
        self.frame._handle_key(PressedKey.ArrowDown, "")
        self.assertEqual(self.frame.side_idx_by_tab["Network"], 2)

        self.frame.focus_depth = 0
        self.frame._handle_key(PressedKey.ArrowRight, "")
        self.assertEqual(self.frame._active_tab(), "Models")
        self.assertEqual(self.frame.side_idx_by_tab["Models"], 0)

        self.frame.focus_depth = 0
        self.frame._handle_key(PressedKey.ArrowLeft, "")
        self.assertEqual(self.frame._active_tab(), "Network")
        self.assertEqual(self.frame.side_idx_by_tab["Network"], 2)


if __name__ == "__main__":
    unittest.main()