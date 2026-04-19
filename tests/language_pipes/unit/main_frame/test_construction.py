import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.content_loader import ContentLoader
from language_pipes.tui.frame.frame_key_handler import FrameKeyHandler

from tests.language_pipes.unit.main_frame.util import _make_main_frame

class TestMainFrameConstruction(unittest.TestCase):
    """Verify that __init__ assembles the object graph and sets initial state."""

    def setUp(self):
        self.frame = _make_main_frame()

    def test_top_headers_match_class_constant(self):
        from language_pipes.tui.frame.main_frame import MainFrame
        self.assertEqual(self.frame.TOP_HEADERS, MainFrame.TOP_HEADERS)

    def test_side_options_match_class_constant(self):
        from language_pipes.tui.frame.main_frame import MainFrame
        self.assertEqual(self.frame.SIDE_OPTIONS_BY_TAB, MainFrame.SIDE_OPTIONS_BY_TAB)

    def test_nav_state_created(self):
        self.assertIsInstance(self.frame.nav, NavState)

    def test_frame_state_created(self):
        self.assertIsInstance(self.frame.state, FrameState)

    def test_exit_confirm_created(self):
        self.assertIsInstance(self.frame.exit_confirm, ExitConfirm)

    def test_confirm_created(self):
        self.assertIsInstance(self.frame.confirm, Confirm)

    def test_loader_created(self):
        self.assertIsInstance(self.frame.provider, ContentLoader)

    def test_key_handler_created(self):
        self.assertIsInstance(self.frame.key_handler, FrameKeyHandler)

    def test_initial_tab_is_network(self):
        """_init_view sets the active tab to 'Network'."""
        self.assertEqual(self.frame.nav.active_tab(), "Network")

    def test_initial_side_option_is_configure(self):
        """_init_view sets the side-nav selection to 'Configure'."""
        self.assertEqual(self.frame.nav.active_side_option(), "Configure")

    def test_initial_focus_depth_is_content(self):
        """After _init_view -> set_side_nav, focus_depth should be 2 (content)."""
        self.assertEqual(self.frame.nav.focus_depth, 2)

    def test_state_not_running_before_run(self):
        """FrameState.running should be False until run() is called."""
        self.assertFalse(self.frame.state.running)

    def test_providers_none_by_default(self):
        """When no providers are passed, provider.providers should be None."""
        self.assertIsNone(self.frame.provider.providers)

    def test_providers_forwarded_to_loader(self):
        providers = {"fake": lambda: None}
        frame = _make_main_frame(providers=providers)
        self.assertIs(frame.provider.providers, providers)

    def test_exit_confirm_not_open(self):
        self.assertFalse(self.frame.exit_confirm.is_open)

    def test_confirm_not_open(self):
        self.assertFalse(self.frame.confirm.is_open)
