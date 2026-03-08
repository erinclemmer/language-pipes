import os
import sys
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.frame.main_frame import MainFrame
from language_pipes.tui.util.kb_utils import PressedKey
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
        self.assertIn("Provider 'get_network_status' unavailable", self.frame.status_message)
        self.assertEqual(self.frame.status_level, "info")

    def test_q_opens_confirmation_instead_of_immediate_exit(self):
        self.frame._handle_key(PressedKey.Alpha, "q")

        self.assertTrue(self.frame.confirm_escape_open)
        self.assertTrue(self.frame.running)
        self.assertFalse(self.frame.exit_tui)

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


class MainFrameProviderWiringTests(unittest.TestCase):
    def setUp(self):
        self.paint_patch = patch.object(TuiGrid, "paint", return_value=None)
        self.paint_patch.start()

    def tearDown(self):
        self.paint_patch.stop()

    def _new_frame(self, providers):
        frame = MainFrame((80, 24), (0, 0), providers=providers)
        frame.running = True
        for provider in providers.values():
            if hasattr(provider, "reset_mock"):
                provider.reset_mock()
        return frame

    def test_refresh_dispatches_to_network_status_provider(self):
        providers = {
            "get_network_status": Mock(return_value={"status": "ok", "uptime": 12}),
        }
        frame = self._new_frame(providers)

        frame._refresh_current_view()

        providers["get_network_status"].assert_called_once_with()
        self.assertEqual(frame.status_message, "Refreshed Network -> Status")
        self.assertEqual(frame.status_level, "info")

    def test_refresh_dispatches_jobs_with_state_filter(self):
        providers = {
            "list_jobs": Mock(return_value=[{"id": "job-1", "state": "queued"}]),
        }
        frame = self._new_frame(providers)
        frame.active_top_idx = frame.TOP_HEADERS.index("Jobs")
        frame.side_idx_by_tab["Jobs"] = 0

        frame._refresh_current_view()

        providers["list_jobs"].assert_called_once_with(state="queued")
        self.assertEqual(frame.status_message, "Refreshed Jobs -> Queue")

    def test_refresh_dispatches_activity_with_level_filter(self):
        providers = {
            "list_activity": Mock(return_value=[{"event": "start", "level": "event"}]),
        }
        frame = self._new_frame(providers)
        frame.active_top_idx = frame.TOP_HEADERS.index("Activity")
        frame.side_idx_by_tab["Activity"] = 1

        frame._refresh_current_view()

        providers["list_activity"].assert_called_once_with(level="event")
        self.assertEqual(frame.status_message, "Refreshed Activity -> Events")

    def test_refresh_dispatches_network_config_provider_for_configure_section(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "node-a",
                    "network_key": "secret",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
        }
        frame = self._new_frame(providers)
        frame.active_top_idx = frame.TOP_HEADERS.index("Network")
        frame.side_idx_by_tab["Network"] = 2

        frame._refresh_current_view()

        providers["get_network_config"].assert_called_once_with()
        self.assertEqual(frame.status_message, "Refreshed Network -> Configure")
        view_state = frame.view_state_by_section[("Network", "Configure")]
        self.assertEqual(view_state["state"], "ok")
        self.assertIn("Network configuration loaded", view_state["summary"])

    def test_refresh_provider_exception_sets_error_status_without_crash(self):
        providers = {
            "get_network_status": Mock(side_effect=RuntimeError("network offline")),
        }
        frame = self._new_frame(providers)

        frame._refresh_current_view()

        self.assertEqual(frame.status_level, "error")
        self.assertEqual(frame.status_message, "Refresh failed for Network -> Status; check provider")
        view_state = frame.view_state_by_section[("Network", "Status")]
        self.assertEqual(view_state["state"], "error")
        self.assertIn("Provider call failed", view_state["summary"])

    def test_missing_provider_falls_back_to_placeholder(self):
        frame = self._new_frame({})
        frame.active_top_idx = frame.TOP_HEADERS.index("Models")
        frame.side_idx_by_tab["Models"] = 0

        frame._refresh_current_view()

        self.assertEqual(frame.status_level, "info")
        self.assertIn("Provider 'list_models' unavailable", frame.status_message)
        view_state = frame.view_state_by_section[("Models", "Installed")]
        self.assertEqual(view_state["state"], "placeholder")
        self.assertIn("Installed model inventory is not loaded yet", view_state["summary"])


class MainFrameEditWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.paint_patch = patch.object(TuiGrid, "paint", return_value=None)
        self.paint_patch.start()

    def tearDown(self):
        self.paint_patch.stop()

    def _new_frame(self, providers):
        frame = MainFrame((80, 24), (0, 0), providers=providers)
        frame.running = True
        for provider in providers.values():
            if hasattr(provider, "reset_mock"):
                provider.reset_mock()
        return frame

    def _choose_apply(self, frame):
        frame._handle_key(PressedKey.ArrowUp, "")
        frame._handle_key(PressedKey.ArrowUp, "")
        frame._handle_key(PressedKey.Enter, "\n")

    def _choose_discard(self, frame):
        frame._handle_key(PressedKey.ArrowUp, "")
        frame._handle_key(PressedKey.Enter, "\n")

    def test_enter_depth2_on_network_configure_opens_edit_form(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "node-a",
                    "network_key": "secret",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
            "save_network_config": Mock(return_value=None),
        }
        frame = self._new_frame(providers)
        frame.side_idx_by_tab["Network"] = 2
        frame.focus_depth = 2

        frame._handle_key(PressedKey.Enter, "\n")

        self.assertTrue(frame.edit_mode)
        self.assertEqual(frame.edit_form_name, "network_config")
        self.assertEqual(frame.edit_fields[0]["value"], "node-a")

    def test_network_field_validation_produces_form_error_state(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "",
                    "network_key": "k",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
            "save_network_config": Mock(return_value=None),
        }
        frame = self._new_frame(providers)
        frame.side_idx_by_tab["Network"] = 2
        frame.focus_depth = 2
        frame._handle_key(PressedKey.Enter, "\n")

        frame._handle_key(PressedKey.Enter, "\n")

        self.assertTrue(frame.edit_mode)
        form_state = frame._active_form_view_state()
        self.assertEqual(form_state["state"], "form")
        self.assertEqual(form_state["fields"][0]["error"], "node_id is required")
        self.assertEqual(frame.status_level, "error")

    def test_network_save_provider_called_with_payload_on_apply(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "node-a",
                    "network_key": "secret",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
            "save_network_config": Mock(return_value=None),
        }
        frame = self._new_frame(providers)
        frame.side_idx_by_tab["Network"] = 2
        frame.focus_depth = 2
        frame._handle_key(PressedKey.Enter, "\n")

        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        self.assertTrue(frame.edit_confirm.is_open)

        self._choose_apply(frame)

        providers["save_network_config"].assert_called_once_with(
            data={
                "node_id": "node-a",
                "network_key": "secret",
                "aes_key": "secret",
                "bootstrap_address": "127.0.0.1",
                "bootstrap_port": 7000,
            }
        )
        self.assertFalse(frame.edit_mode)
        self.assertEqual(frame.status_message, "Saved Network -> Configure")

    def test_discard_from_edit_confirmation_restores_read_only_mode(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "node-a",
                    "network_key": "secret",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
            "save_network_config": Mock(return_value=None),
        }
        frame = self._new_frame(providers)
        frame.side_idx_by_tab["Network"] = 2
        frame.focus_depth = 2
        frame._handle_key(PressedKey.Enter, "\n")

        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        self.assertTrue(frame.edit_confirm.is_open)

        self._choose_discard(frame)

        self.assertFalse(frame.edit_mode)
        self.assertFalse(frame.edit_confirm.is_open)
        self.assertEqual(frame.status_message, "Discarded edits")
        providers["save_network_config"].assert_not_called()

    def test_network_save_failure_sets_error_status_without_crash(self):
        providers = {
            "get_network_config": Mock(
                return_value={
                    "node_id": "node-a",
                    "network_key": "secret",
                    "bootstrap_address": "127.0.0.1",
                    "bootstrap_port": 7000,
                }
            ),
            "save_network_config": Mock(side_effect=RuntimeError("disk full")),
        }
        frame = self._new_frame(providers)
        frame.side_idx_by_tab["Network"] = 2
        frame.focus_depth = 2
        frame._handle_key(PressedKey.Enter, "\n")

        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        frame._handle_key(PressedKey.Enter, "\n")
        self._choose_apply(frame)

        self.assertTrue(frame.edit_mode)
        self.assertEqual(frame.status_level, "error")
        self.assertIn("Apply failed", frame.status_message)


if __name__ == "__main__":
    unittest.main()