from typing import List

class FrameState:
    running: bool
    exit_tui: bool
    status_message: str
    status_level: str
    _installed_model_ids: List[str]

    def __init__(self):
        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"
        
        self._validation_mode_enabled = False
        self._installed_model_ids = []

    def set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def clear_status(self):
        self.status_message = ""
        self.status_level = "info"

    def startup(self):
        self.running = True
        self.exit_tui = False