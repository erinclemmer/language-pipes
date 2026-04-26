from typing import List

from language_pipes.tui.util.kb_utils import PressedKey

class Alert:
    is_open: bool
    message: str
    
    def __init__(self) -> None:
        self.is_open = False
        self.message = "Apply these changes?"
        
    def open(
        self, 
        message: str
    ) -> None:
        self.is_open = True
        self.message = message

    def close(self) -> None:
        self.is_open = False

    def get_lines(self) -> List[str]:
        lines = [
            self.message,
            "",
            "Esc/Enter: close confirmation"
        ]
        return lines

    def handle_key(self, key: PressedKey) -> str:
        if key == PressedKey.Enter:
            return "confirm"
        if key == PressedKey.Escape:
            return "confirm"
        return "nop"