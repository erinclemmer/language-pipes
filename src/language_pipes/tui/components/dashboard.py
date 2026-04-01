from typing import List, Dict, Callable

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall

class Dashboard:

    def __init__(self, loader: ContentLoader, exit_page: Callable, is_focused: Callable):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        pass

    def get_view(self) -> List[str]:
        lines = ["TEST"]
        return lines
    
    def get_footer(self) -> str:
        return ""