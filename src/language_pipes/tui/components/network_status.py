from typing import List, Optional, Callable
import time

from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.content_provider.network_provider import RouterStatus
from language_pipes.tui.frame.provider_calls import ProviderCall

class NetworkStatus:
    loader: ContentLoader
    status: Optional[RouterStatus]
    exit_page: Callable
    is_focused: Callable

    def __init__(self, loader: ContentLoader, exit_page: Callable, is_focused: Callable):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused
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
            if self.loader.call_provider(ProviderCall.is_port_available, self.config.port):
                self.loader.call_provider(ProviderCall.start_network)
    
    def get_view(self) -> List[str]:
        self.config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        self.status: Optional[RouterStatus] = self.loader.call_provider(ProviderCall.get_network_status)
        l_cursor = "|>" if self.is_focused() else "  "
        r_cursor = "<|" if self.is_focused() else "  "
        btn_text = f" {l_cursor} Start Network Server {r_cursor}"
        if not self.loader.call_provider(ProviderCall.is_port_available, self.config.port):
            btn_text = f"Warning: Can't start server, port {self.config.port} is not available"
        lines = ["[X] Server Stopped", "", btn_text]
        if self.status is not None:
            lines = [
                "[O] Server Running" if self.status.running else "[.] Server Starting",
                f"{self.status.num_peers} peer(s) connected",
                "",
                f" {l_cursor} Stop Server {r_cursor}" if self.status.running else f" {l_cursor} Start Server {r_cursor}",
                "", 
                "Logs:"
            ]

            logs = self.status.logs
            
            if len(logs) > 5:
                logs = logs[-5:]

            for ts, log in logs:
                timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
                lines.append(f"{timestamp} {log}")
        return lines
    
    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"