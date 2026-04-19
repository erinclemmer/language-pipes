import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.tui.components.home_dashboard import Dashboard
from language_pipes.content_provider.provider_calls import ProviderCall
from language_pipes.tui.util.kb_utils import PressedKey


class TestDashboardComponent(unittest.TestCase):
    def _make_dashboard(
        self,
        status,
        *,
        change_nav=None,
        exit_page=None,
        used_ram=4.2,
        total_ram=16.0,
        models_to_load=None,
    ):
        provider = Mock()

        def call_provider(provider_call, *args):
            if provider_call == ProviderCall.get_network_config:
                return SimpleNamespace(node_id="test-node")  # Mock config with node_id
            if provider_call == ProviderCall.get_network_status:
                return status
            if provider_call == ProviderCall.get_used_system_ram:
                return used_ram
            if provider_call == ProviderCall.get_total_system_ram:
                return total_ram
            if provider_call == ProviderCall.get_layer_models:
                return models_to_load or []
            if provider_call == ProviderCall.get_models_status:
                return {}
            return None

        provider.call_provider.side_effect = call_provider
        return Dashboard(
            provider, exit_page or Mock(), lambda: True, change_nav or Mock()
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

    def test_dashboard_renders_system_ram_usage(self):
        dashboard = self._make_dashboard(None, used_ram=5.5, total_ram=32.0)

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("System RAM: 5.5 / 32.0 GB", rendered)

    def test_dashboard_hides_peer_count_when_network_stopped(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=False, num_peers=3))

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Network Server: Off", rendered)
        self.assertNotIn("peer(s) connected", rendered)

    def test_dashboard_renders_start_network_server_option_when_stopped(self):
        dashboard = self._make_dashboard(None)

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Start Network Server", rendered)
        self.assertNotIn("Host Models", rendered)

    def test_dashboard_renders_host_models_option_when_running(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=True))

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Stop Network Server", rendered)
        self.assertIn("Host Models", rendered)

    def test_dashboard_renders_hosted_models_using_hosted_view_format(self):
        dashboard = self._make_dashboard(
            None,
            models_to_load=[
                SimpleNamespace(
                    model_id="org/model-a",
                    load_ends=True,
                    device="cuda:0",
                    max_memory=12,
                ),
                SimpleNamespace(
                    model_id="org/model-b", load_ends=False, device="cpu", max_memory=8
                ),
            ],
        )

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Hosted Models", rendered)
        self.assertIn("org/model-a 12GB + ends cuda:0", rendered)
        self.assertIn("org/model-b 8GB  cpu", rendered)

    def test_dashboard_renders_stop_network_server_option_when_running(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=True))

        view = dashboard.get_view()
        rendered = "\n".join(view)

        self.assertIn("Stop Network Server", rendered)

    def test_dashboard_selection_moves_with_arrow_keys_when_stopped(self):
        dashboard = self._make_dashboard(None)

        dashboard.on_key(PressedKey.ArrowDown, "")
        self.assertEqual(dashboard.selected_idx, 0)

        dashboard.on_key(PressedKey.ArrowUp, "")
        self.assertEqual(dashboard.selected_idx, 0)

    def test_dashboard_selection_moves_with_arrow_keys_when_running(self):
        dashboard = self._make_dashboard(SimpleNamespace(running=True))

        dashboard.on_key(PressedKey.ArrowDown, "")
        self.assertEqual(dashboard.selected_idx, 1)

        dashboard.on_key(PressedKey.ArrowUp, "")
        self.assertEqual(dashboard.selected_idx, 0)

    def test_dashboard_enter_starts_network_from_dashboard(self):
        provider = Mock()
        provider.call_provider.return_value = None
        change_nav = Mock()
        dashboard = Dashboard(provider, Mock(), lambda: True, change_nav)

        dashboard.on_key(PressedKey.Enter, "")

        provider.call_provider.assert_any_call(ProviderCall.get_network_status)
        provider.call_provider.assert_any_call(ProviderCall.start_network)
        change_nav.assert_not_called()

    def test_dashboard_enter_hosts_models_when_running(self):
        change_nav = Mock()
        provider = Mock()

        # Create mock models to load
        mock_models = [
            SimpleNamespace(
                model_id="model1", load_ends=False, device="cpu", max_memory=4.0
            ),
            SimpleNamespace(
                model_id="model2", load_ends=True, device="cuda", max_memory=8.0
            ),
        ]

        def call_provider(provider_call, *args):
            if provider_call == ProviderCall.get_network_status:
                return SimpleNamespace(running=True)
            if provider_call == ProviderCall.get_layer_models:
                return mock_models
            return None

        provider.call_provider = Mock(side_effect=call_provider)

        dashboard = Dashboard(
            provider=provider,
            exit_page=lambda: None,
            is_focused=lambda: True,
            change_nav=change_nav,
        )
        dashboard.selected_idx = 1

        dashboard.on_key(PressedKey.Enter, "")

        # Should call host_model for each model
        self.assertEqual(
            provider.call_provider.call_count, 4
        )  # get_network_status, get_models_to_load, host_model x2

        # Verify host_model was called with each model
        host_model_calls = [
            call
            for call in provider.call_provider.call_args_list
            if call[0][0] == ProviderCall.host_layer_model
        ]
        self.assertEqual(len(host_model_calls), 2)
        self.assertEqual(host_model_calls[0][0][1].model_id, "model1")
        self.assertEqual(host_model_calls[1][0][1].model_id, "model2")

        # Should NOT navigate
        change_nav.assert_not_called()

    def test_dashboard_escape_exits_page(self):
        exit_page = Mock()
        dashboard = self._make_dashboard(None, exit_page=exit_page)

        dashboard.on_key(PressedKey.Escape, "")

        exit_page.assert_called_once()
