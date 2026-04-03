import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.tui.util.kb_utils import PressedKey
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys


class TestNetworkFormPage(unittest.TestCase):
    def _make_providers(self):
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.distributed_state_network.objects.endpoint import Endpoint
        from language_pipes.tui.frame.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="test-node",
            credential_dir="/tmp/creds",
            port=5000,
            network_ip="",
            aes_key="secret",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[Endpoint("192.168.1.1", 5000)],
        )

        return {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
            ProviderCall.get_registered_node_ids: lambda: [],
        }

    def _configure_page(self, providers=None):
        frame = _make_main_frame(providers=providers)
        frame.change_nav("Network", "Configure")
        frame.key_handler.activate_selection()
        return frame

    def test_network_configure_is_routed_page(self):
        frame = self._configure_page(providers=self._make_providers())

        self.assertIs(frame.page_router.get_page(), frame.network_form)
        self.assertFalse(frame.editor.edit_mode)
        self.assertGreater(len(frame.network_form.edit_fields), 0)

    def test_activate_selection_without_providers_keeps_page_out_of_field_editor(self):
        frame = self._configure_page(providers=None)

        self.assertIs(frame.page_router.get_page(), frame.network_form)
        self.assertFalse(frame.editor.edit_mode)
        self.assertFalse(frame.network_form.field_editor_visible)
        self.assertEqual(frame.network_form.edit_fields, [])

    def test_escape_from_configure_page_returns_to_side_nav(self):
        frame = self._configure_page(providers=self._make_providers())

        self.assertEqual(frame.nav.focus_depth, 2)
        _simulate_keys(frame, [(PressedKey.Escape, "")])

        self.assertEqual(frame.nav.focus_depth, 1)
        self.assertFalse(frame.network_form.field_editor_visible)

    def test_page_field_navigation(self):
        frame = self._configure_page(providers=self._make_providers())

        self.assertEqual(frame.network_form.edit_field_idx, 0)
        _simulate_keys(frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(frame.network_form.edit_field_idx, 1)

        _simulate_keys(frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(frame.network_form.edit_field_idx, 2)

        _simulate_keys(frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(frame.network_form.edit_field_idx, 1)

    def test_page_field_idx_clamps_at_zero(self):
        frame = self._configure_page(providers=self._make_providers())

        self.assertEqual(frame.network_form.edit_field_idx, 0)
        _simulate_keys(frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(frame.network_form.edit_field_idx, 0)

    def test_node_id_editor_text_blank_after_escape_and_reenter(self):
        frame = self._configure_page(providers=self._make_providers())
        self.assertEqual(frame.network_form.edit_field_idx, 0)

        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(frame.network_form.field_editor_visible)

        _simulate_keys(frame, [(PressedKey.Enter, "")])
        node_id_editor = frame.network_form.node_id_editor
        self.assertTrue(node_id_editor.registering_node_id)

        _simulate_keys(
            frame,
            [
                (PressedKey.Alpha, "m"),
                (PressedKey.Alpha, "y"),
                (PressedKey.Alpha, "n"),
                (PressedKey.Alpha, "o"),
                (PressedKey.Alpha, "d"),
                (PressedKey.Alpha, "e"),
            ],
        )
        self.assertEqual(node_id_editor.new_node_id, "mynode")

        _simulate_keys(frame, [(PressedKey.Escape, "")])
        self.assertFalse(frame.network_form.field_editor_visible)

        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(frame.network_form.field_editor_visible)

        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(node_id_editor.registering_node_id)
        self.assertEqual(node_id_editor.new_node_id, "")
