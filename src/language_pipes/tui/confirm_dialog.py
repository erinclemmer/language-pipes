"""
ConfirmDialog: manages the exit-confirmation overlay state and rendering.
"""
from typing import List

from language_pipes.tui.kb_utils import PressedKey


class ConfirmDialog:
    OPTIONS: List[str] = ["Return to menu", "Exit TUI", "Cancel"]

    is_open: bool
    choice_idx: int

    def __init__(self) -> None:
        self.is_open = False
        self.choice_idx = 2  # default to "Cancel"

    def open(self) -> None:
        self.is_open = True
        self.choice_idx = 2

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
            "Safe exit confirmation",
            "",
            "Use arrows to choose an option, then press Enter:",
        ]
        for i, opt in enumerate(self.OPTIONS):
            cursor = ">" if i == self.choice_idx else " "
            lines.append(f"{cursor} {opt}")
        lines.extend([
            "",
            "Esc: cancel and continue working in MainFrame",
        ])
        return "\n".join(lines)

    def handle_key(self, key: PressedKey) -> str:
        """
        Process a key press while the dialog is open.

        Returns one of:
          "prev"   – moved selection up
          "next"   – moved selection down
          "confirm"– user pressed Enter; caller should read selected_option()
          "cancel" – user pressed Escape
          "nop"    – key had no effect
        """
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
