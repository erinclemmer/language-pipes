from typing import List, Dict, Any

class FrameState:
    running: bool
    exit_tui: bool
    status_message: str
    status_level: str
    edit_mode: bool
    edit_form_name: str
    edit_field_idx: int
    edit_fields: List[Dict[str, Any]]

    def __init__(self):
        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"
        self.edit_field_idx = 0

        self.edit_mode = False
        self.edit_form_name = ""
        self.edit_fields = []

    def set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def clear_status(self):
        self.status_message = ""
        self.status_level = "info"