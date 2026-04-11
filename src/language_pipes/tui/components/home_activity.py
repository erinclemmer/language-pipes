import time
from typing import Callable, List, Optional, Tuple

from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.content_provider.network_provider import RouterStatus
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.util.kb_utils import PressedKey


class HomeActivity:
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
        lines = ["Logs:", ""]

        logs: List[Tuple[float, str]] = []
        network_status: Optional[RouterStatus] = self.loader.call_provider(ProviderCall.get_network_status)
        if network_status is not None:
            logs.extend(network_status.logs)
        
        logs.extend(self.loader.call_provider(ProviderCall.get_oai_logs))
        
        logs.sort(key=lambda x: x[0])

        if len(logs) > 10:
            logs = logs[-10:]

        for ts, log in logs:
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"{timestamp} {log}")
            
        return lines
    
    def get_footer(self) -> str:
        return ""