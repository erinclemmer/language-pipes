from typing import List, Optional, Callable
import time

from language_pipes.content_provider.content_provider import ContentProvider
from distributed_state_network.objects.config import DSNodeConfig
from ansinout import PressedKey
from language_pipes.content_provider.network_provider import RouterStatus

class NetworkStatus:
    provider: ContentProvider
    status: Optional[RouterStatus]
    exit_page: Callable

    def __init__(self, provider: ContentProvider, exit_page: Callable):
        self.provider = provider
        self.exit_page = exit_page
        self.status = None

    def start(self):
        self.status = self.provider.network_provider.get_network_status()

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self.on_enter()
        if key == PressedKey.Escape:
            self.exit_page()

    def on_enter(self):
        status = self.provider.network_provider.get_network_status()
        if status is not None and status.running:
            self.provider.stop_network()
        else:
            if ContentProvider.is_port_available(self.config.port):
                self.provider.network_provider.start_network()
    
    def get_view(self) -> List[str]:
        self.config: DSNodeConfig = self.provider.network_provider.get_network_config()
        self.status: Optional[RouterStatus] = self.provider.network_provider.get_network_status()
        l_cursor = "|>"
        r_cursor = "<|"
        btn_text = f" {l_cursor} Start Network Server {r_cursor}"
        if not ContentProvider.is_port_available(self.config.port):
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