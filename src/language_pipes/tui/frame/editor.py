from typing import List, Dict, Any, Optional, Callable, Tuple

class Editor:
    edit_mode: bool
    edit_form_name: str
    edit_field_idx: int
    edit_fields: List[Dict[str, Any]]
    _pending_apply: Optional[Callable[[], None]]
    _pending_discard: Optional[Callable[[], None]]

    def __init__(self):
        self.edit_field_idx = 0
        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self._pending_apply = None
        self._pending_discard = None

    def exit_edit_mode(self) -> None:
        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []
        self.edit_field_idx = 0
        self._pending_apply = None
        self._pending_discard = None

    def discard_form(self) -> None:
        self.exit_edit_mode()

    def start_edit_mode(self, form_name: str, edit_fields: List[Dict]):
        self.edit_field_idx = 0
        self.edit_mode = True
        self.edit_form_name = form_name
        self.edit_fields = edit_fields

    def set_apply(self, pending_apply: Optional[Callable]):
        self._pending_apply = pending_apply

    def set_discard(self, pending_discard: Optional[Callable]):
        self._pending_discard = pending_discard

    def get_current_field(self) -> Optional[Tuple[str, str]]:
        if not self.edit_fields:
            return None

        field = self.edit_fields[self.edit_field_idx]
        field_name = str(field.get("name", ""))
        raw = str(field.get("value", "")).strip()
        return field_name, raw
    
    def prev_field(self):
        self.edit_field_idx = max(0, self.edit_field_idx - 1)

    def next_field(self):
        self.edit_field_idx = min(len(self.edit_fields) - 1, self.edit_field_idx + 1)