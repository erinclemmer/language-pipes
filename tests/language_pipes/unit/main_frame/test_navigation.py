import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys

class TestNavigation(unittest.TestCase):
    """Verify keyboard-driven navigation through tabs, side-nav, and content."""

    def setUp(self):
        self.frame = _make_main_frame()

    def test_escape_from_content_to_side_nav(self):
        """Pressing Escape at focus_depth=2 should move to depth 1."""
        self.assertEqual(self.frame.nav.focus_depth, 2)
        _simulate_keys(self.frame, [(PressedKey.Escape, "")])
        self.assertEqual(self.frame.nav.focus_depth, 1)

    def test_escape_from_side_nav_to_top_nav(self):
        """Pressing Escape at depth 1 should move to depth 0."""
        _simulate_keys(self.frame, [(PressedKey.Escape, "")])  # 2 -> 1
        _simulate_keys(self.frame, [(PressedKey.Escape, "")])  # 1 -> 0
        self.assertEqual(self.frame.nav.focus_depth, 0)

    def test_escape_at_top_nav_opens_exit_confirm(self):
        """Pressing Escape at depth 0 should open the exit confirmation."""
        _simulate_keys(self.frame, [
            (PressedKey.Escape, ""),  # 2 -> 1
            (PressedKey.Escape, ""),  # 1 -> 0
            (PressedKey.Escape, ""),  # 0 -> exit confirm
        ])
        self.assertTrue(self.frame.exit_confirm.is_open)

    def test_arrow_right_at_top_nav_changes_tab(self):
        """ArrowRight at depth 0 should cycle to the next tab."""
        # Navigate to top-nav first
        _simulate_keys(self.frame, [
            (PressedKey.Escape, ""),  # 2 -> 1
            (PressedKey.Escape, ""),  # 1 -> 0
        ])
        self.assertEqual(self.frame.nav.active_tab(), "Network")
        _simulate_keys(self.frame, [(PressedKey.ArrowRight, "")])
        self.assertEqual(self.frame.nav.active_tab(), "Models")

    def test_arrow_left_at_top_nav_wraps(self):
        """ArrowLeft at depth 0 on the first tab should wrap to the last tab."""
        _simulate_keys(self.frame, [
            (PressedKey.Escape, ""),
            (PressedKey.Escape, ""),
        ])
        self.assertEqual(self.frame.nav.active_tab(), "Network")
        _simulate_keys(self.frame, [(PressedKey.ArrowLeft, "")])
        self.assertEqual(self.frame.nav.active_tab(), "Activity")

    def test_enter_from_top_nav_to_side_nav(self):
        """Enter at depth 0 should move focus to depth 1."""
        _simulate_keys(self.frame, [
            (PressedKey.Escape, ""),
            (PressedKey.Escape, ""),
        ])
        self.assertEqual(self.frame.nav.focus_depth, 0)
        _simulate_keys(self.frame, [(PressedKey.Enter, "")])
        self.assertEqual(self.frame.nav.focus_depth, 1)

    def test_arrow_down_at_side_nav_changes_section(self):
        """ArrowDown at depth 1 should cycle through side-nav options."""
        _simulate_keys(self.frame, [(PressedKey.Escape, "")])  # 2 -> 1
        self.assertEqual(self.frame.nav.focus_depth, 1)
        initial_option = self.frame.nav.active_side_option()
        _simulate_keys(self.frame, [(PressedKey.ArrowDown, "")])
        new_option = self.frame.nav.active_side_option()
        # Should have moved (wraps if at end)
        self.assertNotEqual(initial_option, new_option)

    def test_arrow_up_at_side_nav_changes_section(self):
        """ArrowUp at depth 1 should cycle through side-nav options."""
        _simulate_keys(self.frame, [(PressedKey.Escape, "")])  # 2 -> 1
        initial_idx = self.frame.nav.active_side_idx()
        _simulate_keys(self.frame, [(PressedKey.ArrowUp, "")])
        new_idx = self.frame.nav.active_side_idx()
        self.assertNotEqual(initial_idx, new_idx)

    def test_content_cursor_down(self):
        """ArrowDown at depth 2 should increment content_cursor_idx."""
        self.assertEqual(self.frame.nav.focus_depth, 2)
        initial = self.frame.nav.content_cursor_idx
        _simulate_keys(self.frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(self.frame.nav.content_cursor_idx, initial + 1)

    def test_content_cursor_up_clamps_at_zero(self):
        """ArrowUp at depth 2 with cursor at 0 should stay at 0."""
        self.assertEqual(self.frame.nav.content_cursor_idx, 0)
        _simulate_keys(self.frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(self.frame.nav.content_cursor_idx, 0)

    def test_q_key_opens_exit_confirm(self):
        """Pressing 'q' at any non-overlay state should open exit confirm."""
        _simulate_keys(self.frame, [(PressedKey.Alpha, "q")])
        self.assertTrue(self.frame.exit_confirm.is_open)

    def test_r_key_refreshes_view(self):
        """Pressing 'r' should trigger a content refresh (invalidate cache)."""
        # Pre-populate cache
        self.frame.loader._cache[("Network", "Configure")] = {"state": "cached"}
        _simulate_keys(self.frame, [(PressedKey.Alpha, "r")])
        # After refresh, the cache entry should be updated (force=True replaces it)
        cached = self.frame.loader._cache.get(("Network", "Configure"))
        # The refreshed value should no longer be our injected one
        if cached is not None:
            self.assertNotEqual(cached.get("state"), "cached")

    def test_tab_cycling_wraps_forward(self):
        """Cycling through all tabs with ArrowRight should wrap back to Network."""
        _simulate_keys(self.frame, [
            (PressedKey.Escape, ""),
            (PressedKey.Escape, ""),
        ])
        num_tabs = len(self.frame.TOP_HEADERS)
        keys = [(PressedKey.ArrowRight, "")] * num_tabs
        _simulate_keys(self.frame, keys)
        self.assertEqual(self.frame.nav.active_tab(), "Network")
