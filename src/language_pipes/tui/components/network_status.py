from typing import List, Optional, Callable

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.content_provider import RouterStatus
from language_pipes.tui.frame.provider_calls import ProviderCall

class NetworkStatus:
    loader: ContentLoader
    status: Optional[RouterStatus]
    exit_page: Callable

    def __init__(self, loader: ContentLoader, exit_page: Callable):
        self.loader = loader
        self.exit_page = exit_page
        self.status = None

    def start(self):
        self.status = self.loader.call_provider(ProviderCall.get_network_status)

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self.on_enter()
        if key == PressedKey.Escape:
            self.exit_page()

    def on_enter(self):
        status: RouterStatus = self.loader.call_provider(ProviderCall.get_network_status)
        if status is not None and status.running:
            self.loader.call_provider(ProviderCall.stop_network)
        else:
            self.loader.call_provider(ProviderCall.start_network)
    
    def get_view(self) -> List[str]:
        self.status = self.loader.call_provider(ProviderCall.get_network_status)
        lines = ["[X] Server Stopped", "", " |> Start Network Server <|"]
        if self.status is not None:
            lines = [
                "[O] Server Running" if self.status.running else "[.] Server Starting",
                f"{self.status.num_peers} peer(s) connected",
                "",
                " |> Stop Server <|" if self.status.running else " |> Start Server <|",
                "", 
                "Logs:"
            ]
            lines.extend(self.status.logs[-5:] if len(self.status.logs) > 5 else self.status.logs)
        return lines
    
    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"