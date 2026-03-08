"""
EditConfirmDialog: reusable Apply/Discard/Cancel confirmation overlay.
"""
from typing import List

from language_pipes.tui.util.kb_utils import PressedKey


class EditConfirmDialog:
    OPTIONS: List[str] = ["Apply", "Discard", "Cancel"]

    is_open: bool
    choice_idx: int
    message: str

    def __init__(self) -> None:
        self.is_open = False
        self.choice_idx = 2  # default "Cancel"
        self.message = "Apply these changes?"

    def open(self, message: str) -> None:
        self.is_open = True
        self.choice_idx = 2
        self.message = message

    def close(self) -> None:
        self.is_open = False

    def move_prev(self) -> None:
        self.choice_idx = (self.choice_idx - 1) % len(self.OPTIONS)

    def move_next(self) -> None:
        self.choice_idx = (self.choice_idx + 1) % len(self.OPTIONS)

    def selected_option(self) -> str:
        return self.OPTIONS[self.choice_idx]

    def render(self) -> str:
        lines = [
            "Confirm edit action",
            "",
            self.message,
            "",
            "Use arrows to choose an option, then press Enter:",
        ]
        for i, opt in enumerate(self.OPTIONS):
            cursor = ">" if i == self.choice_idx else " "
            lines.append(f"{cursor} {opt}")
        lines.extend([
            "",
            "Esc: close confirmation and continue editing",
        ])
        return "\n".join(lines)

    def handle_key(self, key: PressedKey) -> str:
        if key in (PressedKey.ArrowUp, PressedKey.ArrowLeft):
            self.move_prev()
            return "prev"
        if key in (PressedKey.ArrowDown, PressedKey.ArrowRight):
            self.move_next()
            return "next"
        if key == PressedKey.Enter:
            return "confirm"
        if key == PressedKey.Escape:
            return "cancel"
        return "nop"
