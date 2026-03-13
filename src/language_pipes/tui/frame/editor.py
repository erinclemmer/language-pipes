from typing import List, Dict, Any, Optional, Callable

from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.frame_state import FrameState

class Editor:
    state: FrameState
    loader: ContentLoader
    edit_mode: bool
    edit_form_name: str
    edit_field_idx: int
    edit_fields: List[Dict[str, Any]]
    _pending_apply: Optional[Callable[[], None]]
    _pending_discard: Optional[Callable[[], None]]

    def __init__(self, state: FrameState, loader: ContentLoader):
        self.state = state
        self.content_loader = loader
        self.edit_field_idx = 0
        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self._pending_apply = None
        self._pending_discard = None

    def _exit_edit_mode(self) -> None:
        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self.edit_field_idx = 0
        self._pending_apply = None
        self._pending_discard = None

    def start_edit_mode(self, form_name: str, edit_fields: List[Dict]):
        self.edit_field_idx = 0
        self.edit_mode = True
        self.edit_form_name = form_name
        self.edit_fields = edit_fields

    def set_apply(self, pending_apply: Optional[Callable]):
        self._pending_apply = pending_apply

    def set_discard(self, pending_discard: Optional[Callable]):
        self._pending_discard = pending_discard

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

        field["error"] = error
        return error is None
    
    def on_form_enter(self, payload: Dict[str, Any]) -> None:
        if not self._validate_current_field():
            self.state.set_status("Fix validation error before continuing", "error")
            return

        if self.edit_field_idx < len(self.edit_fields) - 1:
            self.edit_field_idx += 1
            self.state.set_status("Field accepted", "info")
            return

        if self.edit_form_name == "network_config":
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