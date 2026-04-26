from typing import List

from language_pipes.tui.util.kb_utils import PressedKey

# TODO: Maybe limit alert time to 5 seconds
class Alert:
    is_open: bool
    messages: List[str]
    
    def __init__(self) -> None:
        self.is_open = False
        self.messages = []

    def close(self) -> None:
        self.is_open = False

    def get_lines(self) -> List[str]:
        if len(self.messages) == 0:
            return []
        
        lines = [
            "Alert:",
            self.messages[0],
            "",
            "Esc/Enter: close alert"
        ]
        return lines

    def handle_key(self, key: PressedKey):
        if key == PressedKey.Enter or key == PressedKey.Escape:
            self.messages.pop(0)
            if len(self.messages) == 0:
                self.is_open = False
    
    def create_alert(self, message: str):
        self.is_open = True
        self.messages.append(message)