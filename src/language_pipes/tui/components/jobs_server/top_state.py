from ansinout import PressedKey

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.job_provider import DEFAULT_JOB_PORT
from language_pipes.tui.components.page import PageState
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class TopPageState(PageState):
    focus_idx: int
    server_running: bool
    job_port: int | None
    edit_job_port: str | None
    max_node_jobs: int | None
    edit_max_node_jobs: str | None
    max_api_jobs: int | None
    edit_max_api_jobs: str | None

    def __init__(self):
        super().__init__('top')
        self.focus_idx = 0
        self.server_running = False
        self.job_port = None
        self.edit_job_port = None
        self.max_node_jobs = None
        self.edit_max_node_jobs = None
        self.max_api_jobs = None
        self.edit_max_api_jobs = None

    def on_change(self, args: dict):
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
        elif self.focus_idx == 1:
            self.edit_max_node_jobs = self._max_node_jobs_str() + ch
            self._save_max_node_jobs()
        elif self.focus_idx == 2:
            self.edit_max_api_jobs = self._max_api_jobs_str() + ch
            self._save_max_api_jobs()

    def _on_backspace(self):
        if self.focus_idx == 0:
            self.edit_job_port = self._port_str()[:-1]
        elif self.focus_idx == 1:
            self.edit_max_node_jobs = self._max_node_jobs_str()[:-1]
            self._save_max_node_jobs()
        elif self.focus_idx == 2:
            self.edit_max_api_jobs = self._max_api_jobs_str()[:-1]
            self._save_max_api_jobs()

    def _save_max_node_jobs(self):
        if self._valid_max_node_jobs():
            self.max_node_jobs = int(self.edit_max_node_jobs) # pyright: ignore[reportArgumentType]
            self.provider.job_provider.set_max_node_jobs(self.max_node_jobs)

    def _save_max_api_jobs(self):
        if self._valid_max_api_jobs():
            self.max_api_jobs = int(self.edit_max_api_jobs) # pyright: ignore[reportArgumentType]
            self.provider.job_provider.set_max_api_jobs(self.max_api_jobs)

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        # The server can also be started/stopped from the Home dashboard, which
        # doesn't touch focus_idx, so stopping is keyed off server_running
        # rather than assuming focus landed on the start/stop row.
        if self.server_running:
            self.provider.job_provider.stop_oai_server()
            return

        if self.focus_idx == 0:
            self.focus_idx = 1
        elif self.focus_idx == 1:
            self.focus_idx = 2
        elif self.focus_idx == 2:
            self.focus_idx = 3
        elif self.focus_idx == 3:
            self.change_state('keys', { })
        elif self.focus_idx == 4:
            self._save_and_run()

    def _on_prev(self):
        if not self.server_running:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                self.focus_idx = 4 if self.can_start_server() else 3

    def _on_next(self):
        if not self.server_running:
            self.focus_idx += 1
            max_idx = 4 if self.can_start_server() else 3
            if self.focus_idx > max_idx:
                self.focus_idx = 0

    def _save_and_run(self):
        if not self.validate_job_port():
            return

        self.provider.job_provider.set_job_port(int(self._port_str()))
        self.provider.job_provider.set_api_keys(self.provider.job_provider.get_api_keys())
        self.provider.job_provider.start_oai_server()

    def get_view(self) -> list[str]:
        port_cursor = "|" if self.focus_idx == 0 else ""
        self.server_running = self.provider.job_provider.oai_server_running()
        port_str = self._port_str()
        port_string = f"   Port: {port_str}{port_cursor}" if not self.server_running else f"   Running Server on port {port_str}"

        node_jobs_cursor = "|" if self.focus_idx == 1 else ""
        api_jobs_cursor = "|" if self.focus_idx == 2 else ""

        lines = [
            "Jobs Server:", "",
            port_string
        ]

        if not self.validate_job_port():
            lines.append("   Error: Invalid port value")

        lines.append(f"   Max Node Jobs: {self._max_node_jobs_str()}{node_jobs_cursor}")
        if not self._valid_max_node_jobs():
            lines.append("   Error: Invalid max node jobs value")

        lines.append(f"   Max API Jobs: {self._max_api_jobs_str()}{api_jobs_cursor}")
        if not self._valid_max_api_jobs():
            lines.append("   Error: Invalid max api jobs value")

        api_keys = self.provider.job_provider.get_api_keys()
        lines.append(make_selectable_text(f"{len(api_keys)} api key(s)", self.focus_idx == 3))
        if len(api_keys) == 0:
            lines.extend(["   INFO: No API keys set, authentication not required", ""])

        if not self._network_running():
            lines.append("   WARNING: Network must be connected before server can start")

        if self.server_running:
            lines.extend(["   INFO: Stop server to edit port, job limits, and API Keys", ""])

        if self.can_start_server():
            lines.append(make_selectable_text("Save and Start Server", self.focus_idx == 4))
        elif not self.server_running and not ContentProvider.is_port_available(self._current_port()):
            lines.append(f"   Warning: Can't start server, port {self._current_port()} is not available")

        if self.server_running:
            lines.append(make_selectable_text("Stop Server", True))

        lines.extend(self._get_tip_lines())

        return lines

    def _get_tip_lines(self) -> list[str]:
        tip_key = None
        if self.focus_idx == 0:
            tip_key = "port"
        elif self.focus_idx == 1:
            tip_key = "max_node_jobs"
        elif self.focus_idx == 2:
            tip_key = "max_api_jobs"
        elif self.focus_idx == 3:
            tip_key = "api_keys"

        if tip_key is not None:
            return ["", "Tip:", TIPS["jobs_server"][tip_key]]
        return []

    def get_footer(self) -> str:
        if self.server_running:
            return make_footer_text(["Enter: Stop Server", "Esc: Menu"])

        if self.focus_idx == 0:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type Port", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == 1:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == 2:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == 3:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change", "Esc: Menu"])

        if self.focus_idx == 4:
            return make_footer_text(["Arrows U/D: Move", "Enter: Start Server", "Esc: Menu"])

        return ""

    def _get_job_port(self) -> int | None:
        if self.job_port is None:
            self.job_port = self.provider.job_provider.get_job_port()
        return self.job_port

    def _port_str(self) -> str:
        if self.edit_job_port is None:
            port = self._get_job_port()
            # New/unset config: suggest a default rather than showing a blank field.
            # Nothing is persisted until the user actually saves/starts the server.
            self.edit_job_port = str(port) if port is not None else str(DEFAULT_JOB_PORT)
        return self.edit_job_port

    def _get_max_node_jobs(self) -> int:
        if self.max_node_jobs is None:
            self.max_node_jobs = self.provider.job_provider.get_max_node_jobs()
        return self.max_node_jobs

    def _max_node_jobs_str(self) -> str:
        if self.edit_max_node_jobs is None:
            self.edit_max_node_jobs = str(self._get_max_node_jobs())
        return self.edit_max_node_jobs

    def _valid_max_node_jobs(self) -> bool:
        try:
            return int(self._max_node_jobs_str()) > 0
        except ValueError:
            return False

    def _get_max_api_jobs(self) -> int:
        if self.max_api_jobs is None:
            self.max_api_jobs = self.provider.job_provider.get_max_api_jobs()
        return self.max_api_jobs

    def _max_api_jobs_str(self) -> str:
        if self.edit_max_api_jobs is None:
            self.edit_max_api_jobs = str(self._get_max_api_jobs())
        return self.edit_max_api_jobs

    def _valid_max_api_jobs(self) -> bool:
        try:
            return int(self._max_api_jobs_str()) > 0
        except ValueError:
            return False

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running

    def can_start_server(self) -> bool:
        return self.validate_job_port() and ContentProvider.is_port_available(self._current_port()) and self._network_running() and not self.server_running

    def _current_port(self) -> int | None:
        """The port currently shown/typed in the edit field, not the last-saved config value."""
        try:
            return int(self._port_str())
        except ValueError:
            return None

    def validate_job_port(self) -> bool:
        port = self._current_port()
        return port is not None and port < 65000 and port > 0
