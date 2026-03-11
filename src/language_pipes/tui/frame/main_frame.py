from typing import Any, Callable, Dict, List, Optional, Tuple

from language_pipes.tui.tui import TuiWindow
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.util.kb_utils import PressedKey, read_key
from language_pipes.tui.components.confirm_dialog import ConfirmDialog
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.tui.components.edit_confirm_dialog import EditConfirmDialog

class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Assignments", "Validation"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    def __init__(
        self,
        size: Tuple[int, int],
        pos: Tuple[int, int],
        providers: Optional[object] = None,
    ):
        self.window = TuiWindow(size, pos)

        self.nav = NavState(self.TOP_HEADERS, self.SIDE_OPTIONS_BY_TAB)
        self.confirm = ConfirmDialog()
        self.edit_confirm = EditConfirmDialog()
        self.state = FrameState()
        self.loader = ContentLoader(providers)
        self.layout = FrameLayout(self.window, self.nav, self.loader, self.confirm, self.edit_confirm, self.state)

        self.layout._init_layout(size, pos)
        self._init_view()
        self.layout._render_all()

    def _init_view(self):
        self.nav.set_tab("Network")
        self.nav.set_side_nav(self.layout.side_nav, "Configure")
        self._activate_selection()

    def _activate_selection(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Network" and section == "Configure":
            self._start_network_form()
        elif tab == "Models" and section == "Assignments":
            self._start_model_assignments_form()

    def _start_network_form(self) -> None:
        if not self.loader.provider_available(ProviderCall.get_network_config):
            self.state.set_status("Provider 'get_network_config' unavailable; edit disabled", "info")
            return
        if not self.loader.provider_available(ProviderCall.save_network_config):
            self.state.set_status("Provider 'save_network_config' unavailable; edit disabled", "info")
            return

        try:
            cfg = self.loader.get_network_config()
        except Exception as ex:
            self.state.set_status(f"Failed to load network config: {ex}", "error")
            return

        bootstrap_address = cfg.bootstrap_nodes[0].address if len(cfg.bootstrap_nodes) > 0 else ""
        bootstrap_port = str(cfg.bootstrap_nodes[0].port) if len(cfg.bootstrap_nodes) > 0 else ""

        self.state.start_edit_mode(
            form_name="network_config",
            edit_fields=[
                {"name": "node_id", "value": str(cfg.node_id), "error": None},
                {"name": "network_key", "value": str(cfg.aes_key), "error": None, "masked": True},
                {
                    "name": "bootstrap_address",
                    "value": bootstrap_address,
                    "error": None,
                },
                {"name": "bootstrap_port", "value": str(bootstrap_port), "error": None},
            ]
        )
        
        self.state.set_status("Editing Network -> Configure", "info")

    def _start_model_assignments_form(self) -> None:
        if not self.loader.provider_available(ProviderCall.list_models):
            self.state.set_status("Provider 'list_models' unavailable; edit disabled", "info")
            return
        if not self.loader.provider_available(ProviderCall.save_model_assignments):
            self.state.set_status("Provider 'save_model_assignments' unavailable; edit disabled", "info")
            return

        try:
            model_payload = self.loader.call_provider(ProviderCall.list_models)
            if not isinstance(model_payload, list):
                raise ValueError("list_models must return a list")
        except Exception as ex:
            self.state.set_status(f"Failed to load model assignments: {ex}", "error")
            return

        # TODO

        self.state.set_status("Editing Models -> Assignments", "info")

    def _resolve_edit_confirm_choice(self) -> bool:
        res, msg, lvl = self.edit_confirm.resolve()
        self.state.set_status(msg, lvl)
        return res

    def _handle_edit_confirm_key(self, key: PressedKey) -> bool:
        action = self.edit_confirm.handle_key(key)
        if action == "confirm":
            return self._resolve_edit_confirm_choice()
        elif action == "cancel":
            self.edit_confirm.close()
            self.state.set_status("Edit confirmation canceled", "info")
            return False
        return False

    def _validate_current_field(self) -> bool:
        if not self.edit_fields:
            return False

        field = self.edit_fields[self.edit_field_idx]
        field_name = str(field.get("name", ""))
        raw = str(field.get("value", "")).strip()
        error: Optional[str] = None

        if self.edit_form_name == "network_config":
            if field_name in ("node_id") and raw == "":
                error = f"{field_name} is required"
            elif field_name == "bootstrap_port":
                try:
                    value = int(raw)
                    if value < 1 or value > 65535:
                        error = "bootstrap_port must be 1-65535"
                except Exception:
                    error = "bootstrap_port must be an integer"

        if self.edit_form_name == "model_assignments":
            if field_name == "layer_assignments":
                _, parse_error = self._parse_layer_assignments(raw)
                error = parse_error
            elif field_name == "end_model_id":
                if raw == "":
                    error = "end_model_id is required"
                elif self._installed_model_ids and raw not in self._installed_model_ids:
                    error = "end_model_id must be one of installed model IDs"

        field["error"] = error
        return error is None

    def _parse_layer_assignments(self, text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        text = text.strip()
        if text == "":
            return [], None

        entries = [chunk.strip() for chunk in text.split(",") if chunk.strip() != ""]
        parsed: List[Dict[str, Any]] = []
        seen_layers = set()
        for entry in entries:
            if ":" not in entry:
                return [], "Assignments must be formatted as layer:model"
            layer_txt, model_id = entry.split(":", 1)
            layer_txt = layer_txt.strip()
            model_id = model_id.strip()
            if layer_txt == "" or model_id == "":
                return [], "Assignments must include both layer and model_id"
            try:
                layer_idx = int(layer_txt)
            except Exception:
                return [], "Layer index must be an integer"
            if layer_idx < 0:
                return [], "Layer index must be non-negative"
            if layer_idx in seen_layers:
                return [], "Layer index cannot appear more than once"
            seen_layers.add(layer_idx)
            if self._installed_model_ids and model_id not in self._installed_model_ids:
                return [], f"Unknown model_id '{model_id}'"
            parsed.append({"layer_idx": layer_idx, "model_id": model_id})
        return parsed, None

    def _build_network_payload(self) -> Dict[str, Any]:
        values = {str(f.get("name")): str(f.get("value", "")).strip() for f in self.edit_fields}
        return {
            "node_id": values.get("node_id", ""),
            "network_key": values.get("network_key", ""),
            "aes_key": values.get("network_key", ""),
            "bootstrap_address": values.get("bootstrap_address", ""),
            "bootstrap_port": int(values.get("bootstrap_port", "0")),
        }

    def _build_assignments_payload(self) -> Dict[str, Any]:
        values = {str(f.get("name")): str(f.get("value", "")).strip() for f in self.edit_fields}
        layer_assignments, _ = self._parse_layer_assignments(values.get("layer_assignments", ""))
        return {
            "layer_assignments": layer_assignments,
            "end_model_id": values.get("end_model_id", ""),
        }

    def _on_form_enter(self) -> None:
        if not self._validate_current_field():
            self.state.set_status("Fix validation error before continuing", "error")
            return

        if self.edit_field_idx < len(self.edit_fields) - 1:
            self.edit_field_idx += 1
            self.state.set_status("Field accepted", "info")
            return

        if self.edit_form_name == "network_config":
            payload = self._build_network_payload()

            def apply_network() -> None:
                self.loader.save_network_config(payload)
                self._exit_edit_mode()
                self.state.set_status("Saved Network -> Configure", "info")

            self._open_edit_confirm(
                "Apply changes? Network reconnect may take a few seconds.",
                on_apply=apply_network,
                on_discard=self._discard_form,
            )
            return

        if self.edit_form_name == "model_assignments":
            payload = self._build_assignments_payload()

            def apply_assignments() -> None:
                self.loader.save_model_assignments(payload)
                self._exit_edit_mode()
                self.state.set_status("Saved Models -> Assignments", "info")

            self._open_edit_confirm(
                "Apply model assignment changes?",
                on_apply=apply_assignments,
                on_discard=self._discard_form,
            )

    def _discard_form(self) -> None:
        self._exit_edit_mode()
        self.state.set_status("Discarded edits", "info")

    def _handle_edit_mode_key(self, key: PressedKey, ch: str) -> None:
        if key == PressedKey.Escape:
            self._discard_form()
            self.nav.focus_shallower()
            return
        if key == PressedKey.ArrowUp:
            self.edit_field_idx = max(0, self.edit_field_idx - 1)
            return
        if key == PressedKey.ArrowDown:
            self.edit_field_idx = min(len(self.edit_fields) - 1, self.edit_field_idx + 1)
            return
        if key == PressedKey.Enter:
            self._on_form_enter()
            return
        if not self.edit_fields:
            return

        field = self.edit_fields[self.edit_field_idx]
        value = str(field.get("value", ""))
        if key == PressedKey.Alpha:
            field["value"] = value + ch
            field["error"] = None
            return
        if key == PressedKey.Backspace:
            field["value"] = value[:-1]
            field["error"] = None
            return
        if key == PressedKey.Delete:
            field["value"] = ""
            field["error"] = None

    def _open_exit_confirm(self):
        self.confirm.open()
        self.state.set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

    def _resolve_confirm_choice(self):
        choice = self.confirm.selected_option()
        self.confirm.close()

        if choice == "Return to menu":
            self.exit_tui = False
            self.running = False
            return
        if choice == "Exit TUI":
            self.exit_tui = True
            self.running = False
            return

        self.state.set_status("Exit canceled", "info")

    def _handle_confirm_key(self, key: PressedKey):
        action = self.confirm.handle_key(key)
        if action == "confirm":
            self._resolve_confirm_choice()
        elif action == "cancel":
            self.confirm.close()
            self.state.set_status("Exit canceled", "info")

    def _handle_key(self, key: PressedKey, ch: str):
        if self.confirm.is_open:
            self._handle_confirm_key(key)
            return

        if self.edit_confirm.is_open:
            if self._handle_edit_confirm_key(key):
                self.nav.focus_shallower()
            return

        if self.edit_mode:
            self._handle_edit_mode_key(key, ch)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self.layout._refresh_current_view()
            return

        if key == PressedKey.Escape:
            if self.nav.focus_depth > 0:
                self.nav.focus_shallower()
            else:
                self._open_exit_confirm()
            return

        if key == PressedKey.Enter:
            if self.nav.focus_depth < 1:
                self.nav.focus_deeper()
            else:
                self.nav.focus_deeper()
                self._activate_selection()
            return

        if self.nav.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.nav.tab_next()
                self.state.clear_status()
            elif key == PressedKey.ArrowLeft:
                self.nav.tab_prev()
                self.state.clear_status()
            return

        if self.nav.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.nav.side_next(self.layout.side_nav)
                self.state.clear_status()
            elif key == PressedKey.ArrowUp:
                self.nav.side_prev(self.layout.side_nav)
                self.state.clear_status()
            return

        if self.nav.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.nav.content_cursor_down()
                self.state.set_status("Moved selection cursor (placeholder content)", "info")
            elif key == PressedKey.ArrowUp:
                self.nav.content_cursor_up()
                self.state.set_status("Moved selection cursor (placeholder content)", "info")
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self.state.set_status("No horizontal action in placeholder content", "info")

    def run(self) -> str:
        self.running = True
        self.exit_tui = False
        self.layout._render_all()
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self.layout._render_all()

        self.layout._teardown_windows()
        return "exit" if self.exit_tui else "menu"
