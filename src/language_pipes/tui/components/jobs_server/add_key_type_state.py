from typing import Dict, List

from ansinout import PressedKey

from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class AddKeyTypePageState(PageState):
    type_idx: int

    def __init__(self):
        super().__init__('add_key_type')
        self.type_idx = 0

    def on_change(self, args: Dict):
        self.type_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        elif key == PressedKey.ArrowDown:
            self._on_next()
        elif key == PressedKey.Enter:
            self._on_enter()
        elif key == PressedKey.Escape:
            self._on_escape()

    def _on_escape(self):
        self.change_state('keys', { })

    def _on_enter(self):
        if self.type_idx == 0:
            self.change_state('key_gen', { })
        elif self.type_idx == 1:
            self.change_state('type_key', { })
        elif self.type_idx == 2:
            self.change_state('keys', { })

    def _on_prev(self):
        self.type_idx = (self.type_idx - 1) % 3

    def _on_next(self):
        self.type_idx = (self.type_idx + 1) % 3

    def get_view(self) -> List[str]:
        lines = ["Add New Key:", ""]
        for i, opt in enumerate(['Generate', 'Enter Manually', 'Back']):
            lines.append(make_selectable_text(opt, self.type_idx == i))

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select option", "Esc: Back"])
