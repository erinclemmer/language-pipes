import time
from typing import Callable, List, Tuple

from language_pipes.content_provider.content_provider import ContentProvider
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

        logs: List[Tuple[float, str]] = []
        network_status = self.provider.network_provider.get_network_status()
        if network_status is not None:
            logs.extend(network_status.logs)
        
        logs.extend(self.provider.job_provider.get_oai_logs())

        logs.extend(self.provider.model_provider.get_model_manager_logs())
        
        logs.sort(key=lambda x: x[0])

        if len(logs) > 10:
            logs = logs[-10:]

        for ts, log in logs:
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"{timestamp} {log}")
            
        return lines
    
    def get_footer(self) -> str:
        return " Enter: Dashboard                                                 Esc: Menu "