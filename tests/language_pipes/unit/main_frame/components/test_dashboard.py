import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.tui.util.kb_utils import PressedKey
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys


class TestDashboardComponent(unittest.TestCase):
    def setUp(self):
        self.frame = _make_main_frame()
        self.dashboard = self.frame.page_router.dashboard

    def test_dashboard_renders_two_options(self):
        view = self.dashboard.get_view()
        self.assertIn("Start Network", "\n".join(view))
        self.assertIn("Host Models", "\n".join(view))

    def test_dashboard_enter_routes_to_network_status(self):
        _simulate_keys(self.frame, [(PressedKey.Enter, "")])
        self.assertEqual(self.frame.nav.active_tab(), "Network")
        self.assertEqual(self.frame.nav.active_side_option(), "Status")

    def test_dashboard_selection_moves_with_arrow_keys(self):
        _simulate_keys(self.frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(self.dashboard.selected_idx, 1)
        _simulate_keys(self.frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(self.dashboard.selected_idx, 0)

    def test_dashboard_enter_routes_to_models_hosted(self):
        _simulate_keys(self.frame, [(PressedKey.ArrowDown, ""), (PressedKey.Enter, "")])
        self.assertEqual(self.frame.nav.active_tab(), "Models")
        self.assertEqual(self.frame.nav.active_side_option(), "Hosted")
