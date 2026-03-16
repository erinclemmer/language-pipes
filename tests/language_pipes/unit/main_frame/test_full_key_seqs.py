import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys

class TestFullKeySequences(unittest.TestCase):
    """End-to-end key sequence scenarios through MainFrame."""

    def test_navigate_to_models_installed_and_back(self):
        """Navigate: Network -> top-nav -> Models -> side-nav -> Installed -> back to top."""
        frame = _make_main_frame()

        _simulate_keys(frame, [
            (PressedKey.Escape, ""),          # content -> side-nav
            (PressedKey.Escape, ""),          # side-nav -> top-nav
            (PressedKey.ArrowRight, ""),      # Network -> Models
            (PressedKey.Enter, ""),           # top-nav -> side-nav (Models)
        ])
        self.assertEqual(frame.nav.active_tab(), "Models")
        self.assertEqual(frame.nav.focus_depth, 1)

        _simulate_keys(frame, [
            (PressedKey.Enter, ""),           # side-nav -> content
        ])
        self.assertEqual(frame.nav.focus_depth, 2)
        self.assertEqual(frame.nav.active_side_option(), "Installed")

        _simulate_keys(frame, [
            (PressedKey.Escape, ""),          # content -> side-nav
            (PressedKey.Escape, ""),          # side-nav -> top-nav
        ])
        self.assertEqual(frame.nav.focus_depth, 0)

    def test_cycle_all_tabs_and_verify_side_options(self):
        """Cycle through all tabs and verify each has the correct side options."""
        from language_pipes.tui.frame.main_frame import MainFrame
        frame = _make_main_frame()

        _simulate_keys(frame, [
            (PressedKey.Escape, ""),
            (PressedKey.Escape, ""),
        ])

        for i, tab_name in enumerate(MainFrame.TOP_HEADERS):
            self.assertEqual(frame.nav.active_tab(), tab_name)
            expected_options = MainFrame.SIDE_OPTIONS_BY_TAB[tab_name]
            self.assertEqual(frame.nav.active_side_options(), expected_options)
            _simulate_keys(frame, [(PressedKey.ArrowRight, "")])

    def test_status_message_set_on_exit_confirm_open(self):
        """Opening exit confirm should set a status message."""
        frame = _make_main_frame()
        _simulate_keys(frame, [(PressedKey.Alpha, "q")])
        self.assertIn("Choose", frame.state.status_message)

    def test_multiple_content_cursor_movements(self):
        """Multiple ArrowDown presses should increment content cursor."""
        frame = _make_main_frame()
        for i in range(5):
            _simulate_keys(frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(frame.nav.content_cursor_idx, 5)
