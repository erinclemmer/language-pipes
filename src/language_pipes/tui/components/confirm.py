from enum import Enum
from typing import List, Callable, Optional

from language_pipes.tui.util.kb_utils import PressedKey

class ConfirmState(Enum):
    PROMPT = 0
    CONFIRM = 1
    CANCEL = 2

class Confirm:
    OPTIONS: List[str] = ["Confirm", "Cancel"]

    is_open: bool
    choice_idx: int
    message: str
    confirm_msg: Optional[str]
    cancel_msg: Optional[str]
    on_apply: Optional[Callable]
    on_discard: Optional[Callable]

    def __init__(self) -> None:
        self.is_open = False
        self.choice_idx = 1  # default "Cancel"
        self.message = "Apply these changes?"
        self.on_apply = None
        self.on_discard = None
        self.confirm_msg = None
        self.cancel_msg = None
        self.state = ConfirmState.PROMPT

    def open(
        self, 
        message: str, 
        on_apply: Callable, 
        on_discard: Callable,
        confirm_msg: Optional[str] = None,
        cancel_msg: Optional[str] = None
    ) -> None:
        self.is_open = True
        self.choice_idx = 1
        self.message = message
        self.on_apply = on_apply
        self.on_discard = on_discard
        self.confirm_msg = confirm_msg
        self.cancel_msg = cancel_msg
        self.state = ConfirmState.PROMPT

    def close(self) -> None:
        self.is_open = False

    def move_prev(self) -> None:
        self.choice_idx = (self.choice_idx - 1) % len(self.OPTIONS)

    def move_next(self) -> None:
        self.choice_idx = (self.choice_idx + 1) % len(self.OPTIONS)

    def selected_option(self) -> str:
        return self.OPTIONS[self.choice_idx]

    def render(self) -> str:
        if self.state == ConfirmState.PROMPT:
            return self.get_prompt_lines()
        if self.state == ConfirmState.CONFIRM or self.state == ConfirmState.CANCEL:
            return self.get_confirm_lines()
        return ""

    def get_confirm_lines(self):
        lines = [
            self.confirm_msg if self.state == ConfirmState.CONFIRM else self.cancel_msg,
            "",
            "Esc/Enter: close confirmation"
        ]
        return "\n".join(lines)

    def get_prompt_lines(self) -> str:
        lines = [self.message]
        for i, opt in enumerate(self.OPTIONS):
            cursor = ">" if i == self.choice_idx else " "
            lines.append(f"{cursor} {opt}")
        return "\n".join(lines)

    def handle_key(self, key: PressedKey) -> str:
        if key in (PressedKey.ArrowUp, PressedKey.ArrowLeft):
            self.move_prev()
            return "prev"
        if key in (PressedKey.ArrowDown, PressedKey.ArrowRight):
            self.move_next()
            return "next"
        if key == PressedKey.Enter:
            if self.state == ConfirmState.PROMPT and self.choice_idx == 0 and self.confirm_msg is not None:
                self.state = ConfirmState.CONFIRM
            elif self.state == ConfirmState.PROMPT and self.choice_idx == 1 and self.cancel_msg is not None:
                self.state = ConfirmState.CANCEL
            else:
                if self.choice_idx == 0:
                    return "confirm"
                elif self.choice_idx == 1:
                    return "discard"
        if key == PressedKey.Escape:
            return "discard"
        return "nop"
