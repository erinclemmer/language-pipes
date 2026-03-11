from typing import List, Dict, Any, Optional, Callable

class FrameState:
    running: bool
    exit_tui: bool
    status_message: str
    status_level: str
    edit_mode: bool
    edit_form_name: str
    edit_field_idx: int
    edit_fields: List[Dict[str, Any]]
    _pending_apply: Optional[Callable[[], None]]
    _pending_discard: Optional[Callable[[], None]]
    _installed_model_ids: List[str]

    def __init__(self):
        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"
        self.edit_field_idx = 0

        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self._pending_apply = None
        self._pending_discard = None
        self._validation_mode_enabled = False
        self._installed_model_ids = []

    def set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def clear_status(self):
        self.status_message = ""
        self.status_level = "info"

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