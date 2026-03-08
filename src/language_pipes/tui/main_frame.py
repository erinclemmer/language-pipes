from typing import Any, Callable, Dict, List, Optional, Tuple

from language_pipes.tui.top_nav import TopNav
from language_pipes.tui.side_nav import SideNav
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import PressedKey, read_key
from language_pipes.tui.nav_state import NavState
from language_pipes.tui.confirm_dialog import ConfirmDialog
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.edit_confirm_dialog import EditConfirmDialog
from language_pipes.tui import view_state as vs


class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Assignments", "Validation"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        self.loader = ContentLoader(providers)

        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"

        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields: List[Dict[str, Any]] = []
        self.edit_field_idx = 0
        self._pending_apply: Optional[Callable[[], None]] = None
        self._pending_discard: Optional[Callable[[], None]] = None
        self._validation_mode_enabled = False
        self._installed_model_ids: List[str] = []

        self._init_layout(size, pos)
        self._render_all()

    # ------------------------------------------------------------------
    # Compatibility shims – tests access these attributes directly
    # ------------------------------------------------------------------

    @property
    def focus_depth(self) -> int:
        return self.nav.focus_depth

    @focus_depth.setter
    def focus_depth(self, value: int) -> None:
        self.nav.focus_depth = value

    @property
    def active_top_idx(self) -> int:
        return self.nav.active_top_idx

    @active_top_idx.setter
    def active_top_idx(self, value: int) -> None:
        self.nav.active_top_idx = value

    @property
    def side_idx_by_tab(self) -> Dict[str, int]:
        return self.nav.side_idx_by_tab

    @property
    def confirm_escape_open(self) -> bool:
        return self.confirm.is_open

    @confirm_escape_open.setter
    def confirm_escape_open(self, value: bool) -> None:
        self.confirm.is_open = value

    @property
    def confirm_choice_idx(self) -> int:
        return self.confirm.choice_idx

    @confirm_choice_idx.setter
    def confirm_choice_idx(self, value: int) -> None:
        self.confirm.choice_idx = value

    @property
    def content_cursor_idx(self) -> int:
        return self.nav.content_cursor_idx

    @content_cursor_idx.setter
    def content_cursor_idx(self, value: int) -> None:
        self.nav.content_cursor_idx = value

    @property
    def view_state_by_section(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        # Expose the loader's internal cache for test assertions.
        return self.loader._cache

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 5)), (15, 3))
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))

        self.content_area_size = (max(1, size[0] - 19), max(1, size[1] - 7))
        self.content_bg_id = self.window.add_text(
            TermText(self._content_blank_block()),
            (17, 4),
        )
        self.content_id = self.window.add_text(TermText(""), (17, 4))
        self.footer_id = self.window.add_text(TermText(""), (2, size[1] - 2))

        self.top_nav = TopNav((80, 1), (pos[0], pos[1] + 1), self.TOP_HEADERS)
        self.side_nav = SideNav(
            (13, size[1] - 5),
            (pos[0] + 1, pos[1] + 4),
            self.nav.active_side_options(),
        )

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def _clear_status(self):
        self.status_message = ""
        self.status_level = "info"

    # ------------------------------------------------------------------
    # Navigation helpers (thin wrappers kept for internal use)
    # ------------------------------------------------------------------

    def _active_tab(self) -> str:
        return self.nav.active_tab()

    def _active_side_option(self) -> str:
        return self.nav.active_side_option()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _footer_text(self) -> str:
        if self.confirm.is_open:
            return "Confirm Exit: Arrows U/D or L/R to choose   Enter: Confirm   Esc: Cancel"
        if self.edit_confirm.is_open:
            return "Confirm Edit: Arrows U/D or L/R to choose   Enter: Confirm   Esc: Cancel"
        if self.edit_mode:
            return "Edit Form: Type to edit field   Arrows U/D: Switch field   Enter: Next/Confirm   Esc: Discard"
        if self.nav.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.nav.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return "Arrows U/D: Navigate Placeholder   Enter: Activate   r: Refresh   Esc: Back   q: Exit"

    def _status_text(self) -> str:
        if self.status_message == "":
            return ""
        return f"[{self.status_level.upper()}] {self.status_message}"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_active_view_data(self, update_status: bool, force: bool) -> Dict[str, Any]:
        tab, section = self.nav.active_view_key()
        result = self.loader.load(tab, section, update_status=update_status, force=force)
        if update_status and self.loader.last_status_message:
            self._set_status(self.loader.last_status_message, self.loader.last_status_level)
        return result

    def _refresh_current_view(self):
        self._load_active_view_data(update_status=True, force=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _sync_navigation(self):
        active_options = self.nav.active_side_options()
        self.side_nav.focused_idx = (
            min(self.nav.active_side_idx(), len(active_options) - 1)
            if active_options
            else 0
        )
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.nav.active_top_idx
        interactive_overlay_open = self.confirm.is_open or self.edit_confirm.is_open or self.edit_mode
        self.top_nav.set_focus(self.nav.focus_depth == 0 and not interactive_overlay_open)
        self.side_nav.set_focus(self.nav.focus_depth == 1 and not interactive_overlay_open)

    def _render_content(self):
        self.window.update_text(self.content_bg_id, TermText(self._content_blank_block()))

        if self.confirm.is_open:
            self.window.update_text(self.content_id, TermText(self.confirm.render()))
            return

        if self.edit_confirm.is_open:
            self.window.update_text(self.content_id, TermText(self.edit_confirm.render()))
            return

        if self.edit_mode:
            self._render_form_content()
            return

        tab = self._active_tab()
        section = self._active_side_option()
        view_state = self._load_active_view_data(update_status=False, force=False)

        state_summary = str(view_state.get("summary", "No summary available."))
        next_action = str(view_state.get("hint", "Next: Press r to refresh."))
        level = str(view_state.get("level", "info"))
        details = view_state.get("details", [])

        detail_lines: List[str] = []
        if isinstance(details, list) and details:
            detail_lines.extend([str(line) for line in details])

        selection_hint = f"Selection index: {self.nav.content_cursor_idx + 1}"
        state_label = str(view_state.get("state", "placeholder")).upper()

        content_parts = [
            f"View: {tab}",
            f"Section: {section}",
            "",
            f"State ({state_label}/{level.upper()}): {state_summary}",
        ]

        if detail_lines:
            content_parts.extend(["", "Details:", *detail_lines])

        content_parts.extend([
            "",
            f"Next Action: {next_action}",
            "",
            selection_hint,
            f"Focus depth: {self.nav.focus_depth} (0=top, 1=side, 2=content)",
        ])

        self.window.update_text(self.content_id, TermText("\n".join(content_parts)))

    def _render_form_content(self) -> None:
        tab = self._active_tab()
        section = self._active_side_option()
        view_state = self._active_form_view_state()
        fields = view_state.get("fields", [])
        hint = str(view_state.get("hint", "Complete fields, then confirm."))

        lines = [
            f"View: {tab}",
            f"Section: {section}",
            "",
            f"State (FORM/{str(view_state.get('level', 'info')).upper()}): Edit form active.",
            "",
            "Fields:",
        ]

        for idx, field in enumerate(fields):
            cursor = ">" if idx == self.edit_field_idx else " "
            name = str(field.get("name", "field"))
            value = str(field.get("value", ""))
            error = field.get("error")
            lines.append(f"{cursor} {name}: {value}")
            if error:
                lines.append(f"    ! {error}")

        lines.extend([
            "",
            f"Next Action: {hint}",
            "",
            "Tip: Enter validates current field and advances.",
        ])

        self.window.update_text(self.content_id, TermText("\n".join(lines)))

    def _render_footer(self):
        footer_base = self._footer_text()
        status = self._status_text()
        footer_text = footer_base if status == "" else f"{footer_base}   |   {status}"
        self.window.update_text(self.footer_id, TermText(footer_text))

    def _render_all(self):
        self._sync_navigation()
        self._render_content()
        self._render_footer()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _teardown_windows(self):
        self.window.remove_all()
        self.top_nav.window.remove_all()
        self.side_nav.window.remove_all()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _activate_selection(self):
        tab = self._active_tab()
        section = self._active_side_option()
        if self._is_editable_section(tab, section):
            self._enter_edit_mode(tab, section)
            return

        self._set_status(f"Activated {tab} -> {section} (placeholder action)", "info")

    def _is_editable_section(self, tab: str, section: str) -> bool:
        return (tab, section) in {
            ("Network", "Configure"),
            ("Models", "Assignments"),
            ("Models", "Validation"),
        }

    def _enter_edit_mode(self, tab: str, section: str) -> None:
        if tab == "Network" and section == "Configure":
            self._start_network_form()
            return
        if tab == "Models" and section == "Assignments":
            self._start_model_assignments_form()
            return
        if tab == "Models" and section == "Validation":
            self._toggle_validation_mode()
            return
        self._set_status(f"Edit not available for {tab} -> {section}", "info")

    def _exit_edit_mode(self) -> None:
        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self.edit_field_idx = 0
        self._pending_apply = None
        self._pending_discard = None

    def _start_network_form(self) -> None:
        if not self.loader.provider_available("get_network_config"):
            self._set_status("Provider 'get_network_config' unavailable; edit disabled", "info")
            return
        if not self.loader.provider_available("save_network_config"):
            self._set_status("Provider 'save_network_config' unavailable; edit disabled", "info")
            return

        try:
            cfg = self.loader.get_network_config()
        except Exception as ex:
            self._set_status(f"Failed to load network config: {ex}", "error")
            return

        net_key = cfg.get("network_key")
        if net_key is None:
            net_key = cfg.get("aes_key", "")

        self.edit_form_name = "network_config"
        self.edit_fields = [
            {"name": "node_id", "value": str(cfg.get("node_id", "")), "error": None},
            {"name": "network_key", "value": str(net_key), "error": None, "masked": True},
            {
                "name": "bootstrap_address",
                "value": str(cfg.get("bootstrap_address", "")),
                "error": None,
            },
            {"name": "bootstrap_port", "value": str(cfg.get("bootstrap_port", "")), "error": None},
        ]
        self.edit_field_idx = 0
        self.edit_mode = True
        self._set_status("Editing Network -> Configure", "info")

    def _start_model_assignments_form(self) -> None:
        if not self.loader.provider_available("list_models"):
            self._set_status("Provider 'list_models' unavailable; edit disabled", "info")
            return
        if not self.loader.provider_available("save_model_assignments"):
            self._set_status("Provider 'save_model_assignments' unavailable; edit disabled", "info")
            return

        try:
            model_payload = self.loader.call_provider("list_models")
            if not isinstance(model_payload, list):
                raise ValueError("list_models must return a list")
        except Exception as ex:
            self._set_status(f"Failed to load model assignments: {ex}", "error")
            return

        self._installed_model_ids = self._extract_model_ids(model_payload)

        current_layers: List[Dict[str, Any]] = []
        end_model_id = ""
        for item in model_payload:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("layer_assignments"), list):
                current_layers = item.get("layer_assignments") or []
            if item.get("end_model_id"):
                end_model_id = str(item.get("end_model_id"))

        formatted_layers: List[str] = []
        for assignment in current_layers:
            if not isinstance(assignment, dict):
                continue
            layer_value = assignment.get("layer_idx")
            model_value = assignment.get("model_id")
            if layer_value is None or not model_value:
                continue
            try:
                layer_idx = int(layer_value)
            except Exception:
                continue
            formatted_layers.append(f"{layer_idx}:{str(model_value)}")
        layer_str = ",".join(formatted_layers)

        self.edit_form_name = "model_assignments"
        self.edit_fields = [
            {"name": "layer_assignments", "value": layer_str, "error": None},
            {"name": "end_model_id", "value": end_model_id, "error": None},
        ]
        self.edit_field_idx = 0
        self.edit_mode = True
        self._set_status("Editing Models -> Assignments", "info")

    def _toggle_validation_mode(self) -> None:
        if not self.loader.provider_available("set_validation_mode"):
            self._set_status("Provider 'set_validation_mode' unavailable; toggle disabled", "info")
            return

        current = self._read_validation_mode()
        target = not current
        self._validation_mode_enabled = target
        if not target:
            self._open_edit_confirm(
                "Disable validation mode? This reduces safety checks.",
                on_apply=self._apply_validation_mode,
                on_discard=None,
            )
            self._set_status("Confirm disable for Models -> Validation", "warning")
            return

        try:
            self.loader.set_validation_mode(True)
            self._set_status("Validation mode enabled", "info")
        except Exception as ex:
            self._set_status(f"Failed to set validation mode: {ex}", "error")

    def _read_validation_mode(self) -> bool:
        try:
            if self.loader.provider_available("get_validation_mode"):
                value = self.loader.call_provider("get_validation_mode")
                return bool(value)
            if self.loader.provider_available("get_network_config"):
                cfg = self.loader.get_network_config()
                return bool(cfg.get("model_validation", False))
        except Exception:
            return False
        return False

    def _apply_validation_mode(self) -> None:
        self.loader.set_validation_mode(self._validation_mode_enabled)
        status = "enabled" if self._validation_mode_enabled else "disabled"
        self._set_status(f"Validation mode {status}", "info")

    def _extract_model_ids(self, models: List[Any]) -> List[str]:
        model_ids: List[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            candidate = item.get("model_id") or item.get("id") or item.get("name")
            if candidate is not None:
                text = str(candidate)
                if text not in model_ids:
                    model_ids.append(text)
        return model_ids

    def _active_form_view_state(self) -> Dict[str, Any]:
        fields: List[Dict[str, Any]] = []
        for field in self.edit_fields:
            value = str(field.get("value", ""))
            if field.get("masked"):
                value = "*" * len(value) if value else ""
            fields.append(
                {
                    "name": field.get("name", ""),
                    "value": value,
                    "error": field.get("error"),
                }
            )

        hint = "Complete fields and press Enter on the last field to confirm."
        if self.edit_form_name == "model_assignments":
            hint = "Format layer assignments as layer:model (comma separated), then confirm."
        return vs.form_view_state(fields, hint, "info")

    def _open_edit_confirm(
        self,
        message: str,
        *,
        on_apply: Callable[[], None],
        on_discard: Optional[Callable[[], None]],
    ) -> None:
        self._pending_apply = on_apply
        self._pending_discard = on_discard
        self.edit_confirm.open(message)

    def _resolve_edit_confirm_choice(self) -> None:
        choice = self.edit_confirm.selected_option()
        self.edit_confirm.close()

        if choice == "Apply":
            if self._pending_apply is None:
                self._set_status("No apply action configured", "error")
                return
            try:
                self._pending_apply()
                self._pending_apply = None
                self._pending_discard = None
            except Exception as ex:
                self._set_status(f"Apply failed: {ex}", "error")
            return

        if choice == "Discard":
            if self._pending_discard is not None:
                self._pending_discard()
            else:
                self._set_status("Discarded pending changes", "info")
            self._pending_apply = None
            self._pending_discard = None
            return

        self._set_status("Edit confirmation canceled", "info")

    def _handle_edit_confirm_key(self, key: PressedKey) -> None:
        action = self.edit_confirm.handle_key(key)
        if action == "confirm":
            self._resolve_edit_confirm_choice()
        elif action == "cancel":
            self.edit_confirm.close()
            self._set_status("Edit confirmation canceled", "info")

    def _validate_current_field(self) -> bool:
        if not self.edit_fields:
            return False

        field = self.edit_fields[self.edit_field_idx]
        field_name = str(field.get("name", ""))
        raw = str(field.get("value", "")).strip()
        error: Optional[str] = None

        if self.edit_form_name == "network_config":
            if field_name in ("node_id", "network_key", "bootstrap_address") and raw == "":
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
            self._set_status("Fix validation error before continuing", "error")
            return

        if self.edit_field_idx < len(self.edit_fields) - 1:
            self.edit_field_idx += 1
            self._set_status("Field accepted", "info")
            return

        if self.edit_form_name == "network_config":
            payload = self._build_network_payload()

            def apply_network() -> None:
                self.loader.save_network_config(payload)
                self._exit_edit_mode()
                self._set_status("Saved Network -> Configure", "info")

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
                self._set_status("Saved Models -> Assignments", "info")

            self._open_edit_confirm(
                "Apply model assignment changes?",
                on_apply=apply_assignments,
                on_discard=self._discard_form,
            )

    def _discard_form(self) -> None:
        self._exit_edit_mode()
        self._set_status("Discarded edits", "info")

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
        self._set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

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

        self._set_status("Exit canceled", "info")

    def _handle_confirm_key(self, key: PressedKey):
        action = self.confirm.handle_key(key)
        if action == "confirm":
            self._resolve_confirm_choice()
        elif action == "cancel":
            self.confirm.close()
            self._set_status("Exit canceled", "info")

    def _handle_key(self, key: PressedKey, ch: str):
        if self.confirm.is_open:
            self._handle_confirm_key(key)
            return

        if self.edit_confirm.is_open:
            self._handle_edit_confirm_key(key)
            return

        if self.edit_mode:
            self._handle_edit_mode_key(key, ch)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self._refresh_current_view()
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
                self._clear_status()
            elif key == PressedKey.ArrowLeft:
                self.nav.tab_prev()
                self._clear_status()
            return

        if self.nav.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.nav.side_next(self.side_nav)
                self._clear_status()
            elif key == PressedKey.ArrowUp:
                self.nav.side_prev(self.side_nav)
                self._clear_status()
            return

        if self.nav.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.nav.content_cursor_down()
                self._set_status("Moved selection cursor (placeholder content)", "info")
            elif key == PressedKey.ArrowUp:
                self.nav.content_cursor_up()
                self._set_status("Moved selection cursor (placeholder content)", "info")
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self._set_status("No horizontal action in placeholder content", "info")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> str:
        self.running = True
        self.exit_tui = False
        self._render_all()
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self._render_all()

        self._teardown_windows()
        return "exit" if self.exit_tui else "menu"
