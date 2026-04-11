from typing import Callable, List

from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.util.kb_utils import PressedKey


class DashboardActivity:
    loader: ContentLoader
    exit_page: Callable
    is_focused: Callable[[], bool]

    def __init__(self, loader: ContentLoader, exit_page: Callable, is_focused: Callable):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()
    
    def get_view(self) -> List[str]:
        lines = []
        return lines
    
    def get_footer(self) -> str:
        return ""