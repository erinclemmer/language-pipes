from typing import Dict, List

from ansinout import PressedKey

from language_pipes.tui.components.page import PageState


class TypeKeyPageState(PageState):
    new_api_key: str

    def __init__(self):
        super().__init__('type_key')
        self.new_api_key = ""

    def on_change(self, args: Dict):
        self.new_api_key = ""

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self._on_enter()
        elif key == PressedKey.Escape:
            self._on_escape()
        elif key in (PressedKey.Alpha, PressedKey.Paste):
            self._on_char(ch)
        elif key == PressedKey.Backspace:
            self._on_backspace()

    def _on_char(self, ch: str):
        self.new_api_key += ch

    def _on_backspace(self):
        self.new_api_key = self.new_api_key[:-1]

    def _on_escape(self):
        self.change_state('keys', { })

    def _on_enter(self):
        if len(self.new_api_key) == 0:
            return

        api_keys = self.provider.job_provider.get_api_keys()
        api_keys.append(self.new_api_key)
        self.provider.job_provider.set_api_keys(api_keys)
        self.change_state('keys', { })

    def get_view(self) -> List[str]:
        lines = [
            "Type New API Key:", "",
            f"New Key: {self.new_api_key}|", "",
        ]

        if len(self.new_api_key) == 0:
            lines.append("WARNING: Key is empty, cannot save")

        lines.append("Press Enter to Accept key or Escape to discard key")

        return lines

    def get_footer(self) -> str:
        return ""
