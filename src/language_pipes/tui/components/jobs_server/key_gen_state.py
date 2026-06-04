import secrets
from typing import Dict, List

from ansinout import PressedKey

from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text


class KeyGenPageState(PageState):
    new_api_key: str

    def __init__(self):
        super().__init__('key_gen')
        self.new_api_key = ""

    def on_change(self, args: Dict):
        self.new_api_key = secrets.token_urlsafe(32)

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self._on_enter()
        elif key == PressedKey.Escape:
            self._on_escape()

    def _on_escape(self):
        self.change_state('keys', { })

    def _on_enter(self):
        api_keys = self.provider.job_provider.get_api_keys()
        api_keys.append(self.new_api_key)
        self.provider.job_provider.set_api_keys(api_keys)
        self.change_state('keys', { })

    def get_view(self) -> List[str]:
        return [
            "Generate Key:", "",
            f"New Key: {self.new_api_key}", "",
            "Press Enter to Accept or Escape to go back"
        ]

    def get_footer(self) -> str:
        return make_footer_text(["Enter: Accept key", "Escape: Discard key"])
