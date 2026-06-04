from typing import Dict, List

from ansinout import PressedKey

from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class KeysPageState(PageState):
    key_idx: int
    api_keys: List[str]

    def __init__(self):
        super().__init__('keys')
        self.key_idx = 0
        self.api_keys = []

    def on_change(self, args: Dict):
        self.key_idx = 0
        self.api_keys = self.provider.job_provider.get_api_keys()

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        elif key == PressedKey.ArrowDown:
            self._on_next()
        elif key == PressedKey.Enter:
            self._on_enter()
        elif key == PressedKey.Escape:
            self._on_escape()
        elif key == PressedKey.Delete:
            self._on_delete()

    def _on_delete(self):
        if self.key_idx >= len(self.api_keys):
            return

        key_to_delete = self.api_keys[self.key_idx]

        def on_apply():
            self.api_keys = [key for key in self.api_keys if key != key_to_delete]
            self.provider.job_provider.set_api_keys(self.api_keys)

        self.confirm.open(
            f"Delete {key_to_delete}?",
            on_apply=on_apply,
            on_discard=lambda: None
        )

    def _on_escape(self):
        self.change_state('top', { })

    def _on_enter(self):
        if self.key_idx == len(self.api_keys):
            self.change_state('add_key_type', { })

    def _on_prev(self):
        self.key_idx = (self.key_idx - 1) % (len(self.api_keys) + 1)

    def _on_next(self):
        self.key_idx = (self.key_idx + 1) % (len(self.api_keys) + 1)

    def get_view(self) -> List[str]:
        self.api_keys = self.provider.job_provider.get_api_keys()
        lines = ["API Keys:", ""]

        for i, key in enumerate(self.api_keys):
            lines.append(make_selectable_text(key, self.key_idx == i))

        lines.append("")
        lines.append(make_selectable_text("Add new key", self.key_idx == len(self.api_keys)))

        return lines

    def get_footer(self) -> str:
        if self.key_idx == len(self.api_keys):
            return make_footer_text(["Arrows U/D: Move", "Enter: Add new key", "Esc: Back"])

        return make_footer_text(["Arrows U/D: Move", "Delete: Remove API key", "Esc: Back"])
