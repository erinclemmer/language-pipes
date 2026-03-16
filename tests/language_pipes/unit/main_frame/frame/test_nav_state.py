import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.frame.nav_state import NavState

class TestNavState(unittest.TestCase):
    """Unit tests for NavState in isolation."""

    def setUp(self):
        self.headers = ["A", "B", "C"]
        self.side_opts = {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1", "c2", "c3"]}
        self.nav = NavState(self.headers, self.side_opts)

    def test_initial_tab(self):
        self.assertEqual(self.nav.active_tab(), "A")

    def test_tab_next_wraps(self):
        self.nav.tab_next()
        self.assertEqual(self.nav.active_tab(), "B")
        self.nav.tab_next()
        self.assertEqual(self.nav.active_tab(), "C")
        self.nav.tab_next()
        self.assertEqual(self.nav.active_tab(), "A")

    def test_tab_prev_wraps(self):
        self.nav.tab_prev()
        self.assertEqual(self.nav.active_tab(), "C")

    def test_set_tab(self):
        self.nav.set_tab("B")
        self.assertEqual(self.nav.active_tab(), "B")
        self.assertEqual(self.nav.focus_depth, 1)

    def test_set_tab_invalid_ignored(self):
        self.nav.set_tab("Z")
        self.assertEqual(self.nav.active_tab(), "A")

    def test_active_side_options(self):
        self.assertEqual(self.nav.active_side_options(), ["a1", "a2"])

    def test_active_view_key(self):
        self.assertEqual(self.nav.active_view_key(), ("A", "a1"))

    def test_focus_deeper_clamps(self):
        self.nav.focus_deeper()
        self.nav.focus_deeper()
        self.nav.focus_deeper()
        self.assertEqual(self.nav.focus_depth, 2)

    def test_focus_shallower_clamps(self):
        self.nav.focus_shallower()
        self.assertEqual(self.nav.focus_depth, 0)

    def test_content_cursor_down_up(self):
        self.nav.content_cursor_down()
        self.nav.content_cursor_down()
        self.assertEqual(self.nav.content_cursor_idx, 2)
        self.nav.content_cursor_up()
        self.assertEqual(self.nav.content_cursor_idx, 1)

    def test_content_cursor_up_clamps(self):
        self.nav.content_cursor_up()
        self.assertEqual(self.nav.content_cursor_idx, 0)