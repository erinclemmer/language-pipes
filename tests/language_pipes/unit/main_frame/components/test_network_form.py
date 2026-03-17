import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey
from tests.language_pipes.unit.main_frame.util import _make_main_frame, _simulate_keys

class TestNetworkFormOverlay(unittest.TestCase):
    """Verify that the Network -> Configure edit mode is triggered correctly."""

    def test_activate_selection_starts_network_form_with_providers(self):
        """When providers are available, activate_selection on Network/Configure
        should enter edit mode."""
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

        providers = {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
        }

        frame = _make_main_frame(providers=providers)
        # _init_view already called activate_selection for Network/Configure
        self.assertTrue(frame.editor.edit_mode)
        self.assertEqual(frame.editor.edit_form_name, "network_config")

    def test_activate_selection_without_providers_stays_out_of_edit(self):
        """Without providers, activate_selection should not enter edit mode."""
        frame = _make_main_frame(providers=None)
        # Even though _init_view calls activate_selection, no provider means
        # NetworkForm.start() bails out.
        self.assertFalse(frame.editor.edit_mode)

    def test_escape_from_edit_mode_exits(self):
        """Pressing Escape while in edit mode (not field editor) should exit edit mode."""
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.tui.frame.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="test-node",
            credential_dir="/tmp/creds",
            port=5000,
            network_ip="",
            aes_key="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[],
        )

        providers = {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
        }

        frame = _make_main_frame(providers=providers)
        self.assertTrue(frame.editor.edit_mode)
        _simulate_keys(frame, [(PressedKey.Escape, "")])
        self.assertFalse(frame.editor.edit_mode)

    def test_edit_mode_field_navigation(self):
        """ArrowDown/ArrowUp in edit mode should navigate between fields."""
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.tui.frame.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="test-node",
            credential_dir="/tmp/creds",
            port=5000,
            network_ip="",
            aes_key="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[],
        )

        providers = {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
        }

        frame = _make_main_frame(providers=providers)
        self.assertTrue(frame.editor.edit_mode)
        self.assertEqual(frame.editor.edit_field_idx, 0)

        _simulate_keys(frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(frame.editor.edit_field_idx, 1)

        _simulate_keys(frame, [(PressedKey.ArrowDown, "")])
        self.assertEqual(frame.editor.edit_field_idx, 2)

        _simulate_keys(frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(frame.editor.edit_field_idx, 1)

    def test_edit_mode_field_idx_clamps_at_zero(self):
        """ArrowUp at field index 0 should stay at 0."""
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.tui.frame.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="n",
            credential_dir="/tmp",
            port=5000,
            network_ip="",
            aes_key="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[],
        )

        providers = {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
        }

        frame = _make_main_frame(providers=providers)
        self.assertEqual(frame.editor.edit_field_idx, 0)
        _simulate_keys(frame, [(PressedKey.ArrowUp, "")])
        self.assertEqual(frame.editor.edit_field_idx, 0)

    def test_node_id_editor_text_blank_after_escape_and_reenter(self):
        """Typing in the node-id editor, pressing Escape to return to the form,
        then re-entering the node-id editor should show a blank new_node_id."""
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.tui.frame.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="test-node",
            credential_dir="/tmp/creds",
            port=5000,
            network_ip="",
            aes_key="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[],
        )

        providers = {
            ProviderCall.get_network_config: lambda: fake_config,
            ProviderCall.save_network_config: lambda cfg: None,
            ProviderCall.get_registered_node_ids: lambda: [],
        }

        frame = _make_main_frame(providers=providers)
        self.assertTrue(frame.editor.edit_mode)
        self.assertEqual(frame.editor.edit_field_idx, 0)  # node_id field

        # Enter the node_id field editor
        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(frame.editor.field_editor_visible)

        # Select "Register new node id" (already selected at idx 0 with no existing ids)
        _simulate_keys(frame, [(PressedKey.Enter, "")])
        node_id_editor = frame.network_form.node_id_editor
        self.assertTrue(node_id_editor.registering_node_id)

        # Type some characters
        _simulate_keys(frame, [
            (PressedKey.Alpha, "m"),
            (PressedKey.Alpha, "y"),
            (PressedKey.Alpha, "n"),
            (PressedKey.Alpha, "o"),
            (PressedKey.Alpha, "d"),
            (PressedKey.Alpha, "e"),
        ])
        self.assertEqual(node_id_editor.new_node_id, "mynode")

        # Press Escape to back out to the form
        _simulate_keys(frame, [(PressedKey.Escape, "")])
        # After escape, exit_field_editor -> restart() -> start() re-enters edit mode
        self.assertTrue(frame.editor.edit_mode)
        self.assertFalse(frame.editor.field_editor_visible)

        # Re-enter the node_id field editor
        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(frame.editor.field_editor_visible)

        # Select "Register new node id" again
        _simulate_keys(frame, [(PressedKey.Enter, "")])
        self.assertTrue(node_id_editor.registering_node_id)

        # The text should be blank
        self.assertEqual(node_id_editor.new_node_id, "")
