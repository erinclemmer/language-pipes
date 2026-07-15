import time
from typing import Dict, List, Optional, Tuple

from ansinout import PressedKey

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class TopPageState(PageState):
    focus_idx: int
    server_running: bool
    job_port: Optional[int]
    edit_job_port: Optional[str]

    def __init__(self):
        super().__init__('top')
        self.focus_idx = 0
        self.server_running = False
        self.job_port = None
        self.edit_job_port = None

    def on_change(self, args: Dict):
        self.focus_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        elif key == PressedKey.ArrowDown:
            self._on_next()
        elif key == PressedKey.Enter:
            self._on_enter()
        elif key == PressedKey.Escape:
            self._on_escape()
        elif key in (PressedKey.Alpha, PressedKey.Paste):
            self._on_char(ch)
        elif key == PressedKey.Backspace:
            self._on_backspace()

    def _on_char(self, ch: str):
        if self.focus_idx == 0:
            self.edit_job_port = self._port_str() + ch

    def _on_backspace(self):
        if self.focus_idx == 0:
            self.edit_job_port = self._port_str()[:-1]

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        if self.focus_idx == 0:
            self.focus_idx = 1
        elif self.focus_idx == 1:
            self.change_state('keys', { })
        elif self.focus_idx == 2:
            if self.server_running:
                self.provider.job_provider.stop_oai_server()
            else:
                self._save_and_run()

    def _on_prev(self):
        if not self.server_running:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                self.focus_idx = 2 if self.can_start_server() else 1

    def _on_next(self):
        if not self.server_running:
            self.focus_idx += 1
            max_idx = 2 if self.can_start_server() else 1
            if self.focus_idx > max_idx:
                self.focus_idx = 0

    def _save_and_run(self):
        if not self.validate_job_port():
            return

        self.provider.job_provider.set_job_port(int(self._port_str()))
        self.provider.job_provider.set_api_keys(self.provider.job_provider.get_api_keys())
        self.provider.job_provider.start_oai_server()

    def get_view(self) -> List[str]:
        port_cursor = "|" if self.focus_idx == 0 else ""
        self.server_running = self.provider.job_provider.oai_server_running()
        port_str = self._port_str()
        port_string = f"   Port: {port_str}{port_cursor}" if not self.server_running else f"   Running Server on port {port_str}"
        lines = [
            "Jobs Server:", "",
            port_string
        ]

        if not self.validate_job_port():
            lines.append("   Error: Invalid port value")

        api_keys = self.provider.job_provider.get_api_keys()
        lines.append(make_selectable_text(f"{len(api_keys)} api key(s)", self.focus_idx == 1))
        if len(api_keys) == 0:
            lines.extend(["   INFO: No API keys set, authentication not required", ""])

        if not self._network_running():
            lines.append("   WARNING: Network must be connected before server can start")

        if self.server_running:
            lines.extend(["   INFO: Stop server to edit port and API Keys", ""])

        if self.can_start_server():
            lines.append(make_selectable_text("Start Server", self.focus_idx == 2))
        elif not self.server_running and not ContentProvider.is_port_available(self._get_job_port()):
            lines.append(f"   Warning: Can't start server, port {self._get_job_port()} is not available")

        if self.server_running:
            lines.append(make_selectable_text("Stop Server", True))

        logs: List[Tuple[float, str]] = self.provider.job_provider.get_oai_logs()
        lines.extend(["", "Logs:"])

        if len(logs) > 5:
            logs = logs[-5:]

        for ts, log in logs:
            timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"{timestamp} {log}")

        return lines

    def get_footer(self) -> str:
        if self.server_running:
            return make_footer_text(["Enter: Stop Server", "Esc: Menu"])

        if self.focus_idx == 0:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type Port", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == 1:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change API keys", "Esc: Menu"])

        if self.focus_idx == 2:
            return make_footer_text(["Arrows U/D: Move", "Enter: Start Server", "Esc: Menu"])

        return ""

    def _get_job_port(self) -> int:
        if self.job_port is None:
            self.job_port = self.provider.job_provider.get_job_port()
        return self.job_port

    def _port_str(self) -> str:
        if self.edit_job_port is None:
            self.edit_job_port = str(self._get_job_port())
        return self.edit_job_port

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running

    def can_start_server(self) -> bool:
        return self.validate_job_port() and ContentProvider.is_port_available(self._get_job_port()) and self._network_running() and not self.server_running

    def validate_job_port(self) -> bool:
        try:
            port = int(self._port_str())
            return port < 65000 and port > 0
        except ValueError:
            return False
