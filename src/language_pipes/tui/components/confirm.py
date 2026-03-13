from typing import List, Callable, Optional, Tuple

from language_pipes.tui.util.kb_utils import PressedKey


class Confirm:
    OPTIONS: List[str] = ["Apply", "Discard", "Cancel"]

    is_open: bool
    choice_idx: int
    message: str
    on_apply: Optional[Callable]
    on_discard: Optional[Callable]

    def __init__(self) -> None:
        self.is_open = False
        self.choice_idx = 2  # default "Cancel"
        self.message = "Apply these changes?"

    def open(self, message: str, on_apply: Callable, on_discard: Optional[Callable]) -> None:
        self.is_open = True
        self.choice_idx = 2
        self.message = message
        self.on_apply = on_apply
        self.on_discard = on_discard

    def resolve(self) -> Tuple[bool, str, str]:
        choice = self.selected_option()
        self.close()

        if choice == "Apply":
            if self._pending_apply is None:
                return False, "No apply action configured", "error"
            try:
                self._pending_apply()
                self._pending_apply = None
                self._pending_discard = None
            except Exception as ex:
                return False, f"Apply failed: {ex}", "error"

        if choice == "Discard":
            status = ""
            if self._pending_discard is not None:
                self._pending_discard()
            else:
                status = "Discarded pending changes"
            self._pending_apply = None
            self._pending_discard = None
            return True, status, "info"

        return False, "Edit confirmation canceled", "info"

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
