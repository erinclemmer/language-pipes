from typing import Dict, List

from ansinout import PressedKey

from language_pipes.tui.components.page import PageState
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.text import make_footer_text, make_selectable_text

DATA_TYPES = ["bf16", "int8", "int4"]

class ChooseDataTypePageState(PageState):
    select_idx: int

    def __init__(self):
        super().__init__('choose_data_type')
        self.select_idx = 0

    def on_change(self, args: Dict):
        self.select_idx = 0
        if args.get("data_type") == 8:
            self.select_idx = 1
        if args.get("data_type") == 4:
            self.select_idx = 2

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        if key == PressedKey.ArrowDown:
            self._on_next()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()

    def _on_next(self):
        self.select_idx = (self.select_idx + 1) % len(DATA_TYPES)

    def _on_prev(self):
        self.select_idx = (self.select_idx - 1) % len(DATA_TYPES)

    def _on_enter(self):
        dtype = 16
        if self.select_idx == 1:
            dtype = 8
        if self.select_idx == 2:
            dtype = 4
        self.change_state('edit', { "data_type": dtype })

    def _on_escape(self):
        self.change_state('edit', { })

    def get_view(self) -> List[str]:
        lines = ["Choose Data Type", ""]

        for i, dtype in enumerate(DATA_TYPES):
            lines.extend([make_selectable_text(dtype, self.select_idx == i), ""])

        tip = TIPS["layer_models"][DATA_TYPES[self.select_idx]]
        lines.extend([
            "",
            "Tip:",
            tip
        ])

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Data Type", "Esc: Back"])
