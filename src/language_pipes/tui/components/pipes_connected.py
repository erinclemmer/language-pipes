from typing import List, Callable

from language_pipes.tui.util.kb_utils import PressedKey


class PipesConnected:
    def __init__(
        self,
        loader: object,
        exit_page: Callable,
        is_focused: Callable,
    ):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        return ["Hello, world"]

    def get_footer(self) -> str:
        return "Esc: Back"
