import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.tui.components.dashboard import Dashboard
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.util.kb_utils import PressedKey


class TestDashboardComponent(unittest.TestCase):
    def _make_dashboard(self, status, *, change_nav=None, exit_page=None):
        loader = Mock()
        loader.call_provider.return_value = status
        return Dashboard(
            loader, exit_page or Mock(), lambda: True, change_nav or Mock()
        )

    def test_dashboard_renders_network_on_when_running(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=True, num_peers=0))

        view = dashboard.get_view()

        self.assertEqual(view[0], "Network Server: On (0 peer(s) connected)")

    def test_dashboard_renders_network_off_when_stopped(self):
        dashboard = self._make_dashboard(None)

        view = dashboard.get_view()

        self.assertEqual(view[0], "Network Server: Off")

    def test_dashboard_renders_peer_count_only_when_running(self):
        dashboard = self._make_dashboard(
            SimpleNamespace(running=True, num_peers=3, logs=["a", "b"])
        )

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Network Server: On", rendered)
        self.assertIn("3 peer(s) connected", rendered)
        self.assertNotIn("Logs:", rendered)
        self.assertNotIn("Server Running", rendered)
        self.assertNotIn("Server Stopped", rendered)

    def test_dashboard_hides_peer_count_when_network_stopped(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=False, num_peers=3))

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Network Server: Off", rendered)
        self.assertNotIn("peer(s) connected", rendered)

    def test_dashboard_still_renders_existing_options(self):
        dashboard = self._make_dashboard(None)

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Start Network Server", rendered)
        self.assertIn("Host Models", rendered)

    def test_dashboard_renders_stop_network_server_option_when_running(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=True))

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Stop Network Server", rendered)

    def test_dashboard_selection_moves_with_arrow_keys(self):
        dashboard = self._make_dashboard(None)

        dashboard.on_key(PressedKey.ArrowDown, "")
        self.assertEqual(dashboard.selected_idx, 1)

        dashboard.on_key(PressedKey.ArrowUp, "")
        self.assertEqual(dashboard.selected_idx, 0)

    def test_dashboard_enter_starts_network_from_dashboard(self):
        loader = Mock()
        loader.call_provider.return_value = None
        change_nav = Mock()
        dashboard = Dashboard(loader, Mock(), lambda: True, change_nav)

        dashboard.on_key(PressedKey.Enter, "")

        loader.call_provider.assert_any_call(ProviderCall.get_network_status)
        loader.call_provider.assert_any_call(ProviderCall.start_network)
        change_nav.assert_not_called()

    def test_dashboard_enter_still_routes_host_models(self):
        change_nav = Mock()
        dashboard = self._make_dashboard(None, change_nav=change_nav)
        dashboard.selected_idx = 1

        dashboard.on_key(PressedKey.Enter, "")

        change_nav.assert_called_once_with("Models", "Hosted")

    def test_dashboard_escape_exits_page(self):
        exit_page = Mock()
        dashboard = self._make_dashboard(None, exit_page=exit_page)

        dashboard.on_key(PressedKey.Escape, "")

        exit_page.assert_called_once()
