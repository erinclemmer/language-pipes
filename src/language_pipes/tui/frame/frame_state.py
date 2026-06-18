from typing import List

class FrameState:
    running: bool
    exit_tui: bool
    _installed_model_ids: List[str]

    def __init__(self):
        self.running = False
        self.exit_tui = False
        
        self._validation_mode_enabled = False
        self._installed_model_ids = []

    def startup(self):
        self.running = True
        self.exit_tui = False