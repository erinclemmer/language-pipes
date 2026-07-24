import time
from typing import Callable, List

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.util.logging import get_ring_buffer
from ansinout import PressedKey


class HomeActivity:
    provider: ContentProvider
    exit_page: Callable
    change_nav: Callable[[str, str], None]

    def __init__(self, provider: ContentProvider, exit_page: Callable, change_nav: Callable):
        self.provider = provider
        self.exit_page = exit_page
        self.change_nav = change_nav

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()
        if key == PressedKey.Enter:
            self.change_nav("Home", "Dashboard")
    
    def get_view(self) -> List[str]:
        lines = ["Activity:"]

        logs = get_ring_buffer().get(limit=10)

        for ts, log in logs:
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"{timestamp} {log}")
            
        return lines
    
    def get_footer(self) -> str:
        return " Enter: Dashboard                                                 Esc: Menu "