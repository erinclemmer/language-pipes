from typing import Dict, List

from ansinout import PressedKey

from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text


class ApiKeyPageState(PageState):
    def __init__(self):
        super().__init__('api_key')
        self.token_string = ""

    def on_change(self, args: Dict):
        self.token_string = ""

    def on_key(self, key: PressedKey, ch: str):
        if key in (PressedKey.Alpha, PressedKey.Paste):
            self._on_char(ch)
        if key == PressedKey.Backspace:
            self._on_backspace()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()

    def _on_char(self, ch: str):
        self.token_string += ch

    def _on_backspace(self):
        self.token_string = self.token_string[:-1]

    def _on_enter(self):
        token = self.token_string

        def save_token():
            ModelProvider.save_hf_token(token)
            self.change_state('download', {'token': token})

        def use_without_saving():
            self.change_state('download', {'token': token})

        self.confirm.open(
            f"Save this token?\n{token}",
            on_apply=save_token,
            on_discard=use_without_saving,
        )

    def _on_escape(self):
        self.change_state('download', {})

    def get_view(self) -> List[str]:
        token_string = self.token_string[-40:] if len(self.token_string) > 40 else self.token_string
        return [
            "Type or paste to enter a huggingface API key", "",
            f"API Key |> {token_string}|",
            "",
            "Create an access token at https://huggingface.co/settings/tokens",
        ]

    def get_footer(self) -> str:
        return make_footer_text(["Type: API Key", "Enter: Confirm", "Esc: Back"])
