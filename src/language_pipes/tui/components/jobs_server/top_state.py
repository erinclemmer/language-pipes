from ansinout import PressedKey

from language_pipes.config import default_max_api_jobs, default_max_node_jobs
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.job_provider import DEFAULT_JOB_PORT
from language_pipes.tui.components.page import PageState
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


PORT_IDX = 0
MAX_NODE_JOBS_IDX = 1
MAX_API_JOBS_IDX = 2
MAX_NODE_MEMORY_IDX =3
MAX_CACHE_TIME_IDX = 4
API_KEYS_IDX = 5
SAVE_IDX = 6

class TopPageState(PageState):
    focus_idx: int
    server_running: bool

    job_port: int | None
    max_api_jobs: int | None
    max_node_jobs: int | None
    max_cache_time: int | None
    max_node_memory: float | None

    edit_job_port: str
    edit_max_api_jobs: str
    edit_max_node_jobs: str
    edit_max_cache_time: str
    edit_max_node_memory: str

    def __init__(self, provider: ContentProvider):
        super().__init__('top')
        self.server_running = provider.job_provider.oai_server_running()
        self.focus_idx = PORT_IDX if not self.server_running else SAVE_IDX

        self.job_port = provider.job_provider.get_job_port()
        self.edit_job_port = str(self.job_port) if self.job_port is not None else ""

        self.max_node_jobs = provider.job_provider.get_max_node_jobs()
        self.edit_max_node_jobs = str(self.max_node_jobs) if self.max_node_jobs is not None else ""

        self.max_api_jobs = provider.job_provider.get_max_api_jobs()
        self.edit_max_api_jobs = str(self.max_api_jobs) if self.max_api_jobs is not None else ""

        self.max_node_memory = provider.job_provider.get_max_node_memory()
        self.edit_max_node_memory = str(self.edit_max_node_memory) if self.max_node_memory is not None else ""

        self.max_cache_time = provider.job_provider.get_max_cache_time()
        self.edit_max_cache_time = str(self.edit_max_cache_time) if self.max_cache_time is not None else ""

    def on_change(self, args: dict):
        self.focus_idx = PORT_IDX if not self.provider.job_provider.oai_server_running() else SAVE_IDX

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
        if self.focus_idx == PORT_IDX:
            self.edit_job_port += ch
            self._save_job_port()
        elif self.focus_idx == MAX_NODE_JOBS_IDX:
            self.edit_max_node_jobs += ch
            self._save_max_node_jobs()
        elif self.focus_idx == MAX_API_JOBS_IDX:
            self.edit_max_api_jobs += ch
            self._save_max_api_jobs()
        elif self.focus_idx == MAX_NODE_MEMORY_IDX:
            self.edit_max_node_memory += ch
            self._save_max_node_memory()
        elif self.focus_idx == MAX_CACHE_TIME_IDX:
            self.edit_max_cache_time += ch
            self._save_max_cache_time()

    def _on_backspace(self):
        if self.focus_idx == PORT_IDX:
            if self.edit_job_port is not None:
                self.edit_job_port = self.edit_job_port[:-1]
            self._save_job_port()
        elif self.focus_idx == MAX_NODE_JOBS_IDX:
            if self.edit_max_node_jobs is not None:
                self.edit_max_node_jobs = self.edit_max_node_jobs[:-1]
            self._save_max_node_jobs()
        elif self.focus_idx == MAX_API_JOBS_IDX:
            if self.edit_max_api_jobs is not None:
                self.edit_max_api_jobs = self.edit_max_api_jobs[:-1]
            self._save_max_api_jobs()
        elif self.focus_idx == MAX_NODE_MEMORY_IDX:
            if self.edit_max_node_memory is not None:
                self.edit_max_node_memory = self.edit_max_node_memory[:-1]
            self._save_max_node_memory()
        elif self.focus_idx == MAX_CACHE_TIME_IDX:
            if self.edit_max_cache_time is not None:
                self.edit_max_cache_time = self.edit_max_cache_time[:-1]
            self._save_max_cache_time()

    def _save_job_port(self):
        if self._valid_job_port():
            self.job_port = int(self.edit_job_port) if self.edit_job_port != "" else None
            self.provider.job_provider.set_job_port(self.job_port)

    def _save_max_node_jobs(self):
        if self._valid_max_node_jobs():
            self.max_node_jobs = int(self.edit_max_node_jobs) if self.edit_max_node_jobs != "" else None
            self.provider.job_provider.set_max_node_jobs(self.max_node_jobs)

    def _save_max_api_jobs(self):
        if self._valid_max_api_jobs():
            self.max_api_jobs = int(self.edit_max_api_jobs) if self.edit_max_api_jobs != "" else None
            self.provider.job_provider.set_max_api_jobs(self.max_api_jobs)

    def _save_max_node_memory(self):
        if self._valid_max_node_memory():
            self.max_node_memory = float(self.edit_max_node_memory) if self.edit_max_node_memory != "" else None
            self.provider.job_provider.set_max_node_memory(self.max_node_memory)

    def _save_max_cache_time(self):
        if self._valid_max_cache_time():
            self.max_cache_time = int(self.edit_max_cache_time) if self.edit_max_cache_time != "" else None
            self.provider.job_provider.set_max_cache_time(self.max_cache_time)

    def _valid_job_port(self) -> bool:
        if self.edit_job_port == "":
            return True
        try:
            port = int(self.edit_job_port)
            return port < 65000 and port > 0
        except Exception:
            return False
        
    def _valid_max_api_jobs(self) -> bool:
        if self.edit_max_api_jobs == "":
            return True
        try:
            return int(self.edit_max_api_jobs) > 0
        except ValueError:
            return False

    def _valid_max_node_jobs(self) -> bool:
        if self.edit_max_node_jobs == "":
            return True
        try:
            return int(self.edit_max_node_jobs) > 0
        except ValueError:
            return False

    def _valid_max_node_memory(self) -> bool:
        if self.edit_max_node_memory == "":
            return True
        try:
            return float(self.edit_max_node_memory) > 0
        except Exception:
            return False

    def _valid_max_cache_time(self) -> bool:
        if self.edit_max_cache_time == "":
            return True
        try:
            return int(self.edit_max_cache_time) > 0
        except Exception:
            return False

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        # The server can also be started/stopped from the Home dashboard, which
        # doesn't touch focus_idx, so stopping is keyed off server_running
        # rather than assuming focus landed on the start/stop row.
        if self.server_running:
            self.provider.job_provider.stop_oai_server()
            return

        if self.focus_idx == PORT_IDX:
            self.focus_idx = MAX_NODE_JOBS_IDX
        elif self.focus_idx == MAX_NODE_JOBS_IDX:
            self.focus_idx = MAX_API_JOBS_IDX
        elif self.focus_idx == MAX_API_JOBS_IDX:
            self.focus_idx = MAX_NODE_MEMORY_IDX
        elif self.focus_idx == MAX_NODE_MEMORY_IDX:
            self.focus_idx = API_KEYS_IDX
        elif self.focus_idx == API_KEYS_IDX:
            self.change_state('keys', { })
        elif self.focus_idx == SAVE_IDX:
            self._save_and_run()

    def _on_prev(self):
        if not self.server_running:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                self.focus_idx = SAVE_IDX if self.can_start_server() else API_KEYS_IDX

    def _on_next(self):
        if not self.server_running:
            self.focus_idx += 1
            max_idx = SAVE_IDX if self.can_start_server() else API_KEYS_IDX
            if self.focus_idx > max_idx:
                self.focus_idx = PORT_IDX

    def _save_and_run(self):
        if not self.can_start_server():
            return

        self.provider.job_provider.start_oai_server()

    def get_view(self) -> list[str]:
        port_cursor = "|" if self.focus_idx == PORT_IDX else ""
        self.server_running = self.provider.job_provider.oai_server_running()

        node_jobs_cursor = "|" if self.focus_idx == MAX_NODE_JOBS_IDX else ""
        api_jobs_cursor = "|" if self.focus_idx == MAX_API_JOBS_IDX else ""
        node_memory_cursor = "|" if self.focus_idx == MAX_NODE_MEMORY_IDX else ""
        cache_time_cursor = "|" if self.focus_idx == MAX_CACHE_TIME_IDX else ""

        lines = [
            "Jobs Server:", ""
        ]

        if self.edit_job_port == "":
            lines.append(f"   Port: (Default {DEFAULT_JOB_PORT}){port_cursor}")
        else:
            lines.append(f"   Port: {self.edit_job_port}{port_cursor}")
        if not self._valid_job_port():
            lines.append("   Error: Invalid job port value")

        if self.edit_max_node_jobs == "":
            lines.append(f"   Max Node Jobs: (Default {default_max_node_jobs()}){node_jobs_cursor}")
        else:
            lines.append(f"   Max Node Jobs: {self.edit_max_node_jobs}{node_jobs_cursor}")
        if not self._valid_max_node_jobs():
            lines.append("   Error: Invalid max node jobs value")

        if self.edit_max_api_jobs == "":
            lines.append(f"   Max API Jobs: (Default {default_max_api_jobs()}){api_jobs_cursor}")
        else:
            lines.append(f"   Max API Jobs: {self.edit_max_api_jobs}{api_jobs_cursor}")
        if not self._valid_max_api_jobs():
            lines.append("   Error: Invalid max api jobs value")

        if self.edit_max_node_memory == "":
            lines.append(f"   Max Node Memory: (Default unlimited){node_memory_cursor}")
        else:
            lines.append(f"   Max Node Memory: {self.max_node_memory}{node_memory_cursor} GB")
        if not self._valid_max_node_memory():
            lines.append("   Error: Invalid max node memory value")

        if self.edit_max_cache_time == "":
            lines.append(f"   Max Cache Time: (Default no caching){cache_time_cursor}")
        else:
            lines.append(f"   Max Cache Time: {self.max_cache_time}{cache_time_cursor} minutes")
        if not self._valid_max_cache_time():
            lines.append("   Error: Invalid max cache time value")

        api_keys = self.provider.job_provider.get_api_keys()
        lines.append(make_selectable_text(f"{len(api_keys)} api key(s)", self.focus_idx == API_KEYS_IDX))
        if len(api_keys) == 0:
            lines.extend(["   INFO: No API keys set, authentication not required", ""])

        if not self._network_running():
            lines.append("   WARNING: Network must be connected before server can start")

        if self.server_running:
            lines.extend(["   INFO: Stop server to edit port, job limits, and API Keys", ""])

        if self.can_start_server():
            lines.append(make_selectable_text("Start Server", self.focus_idx == SAVE_IDX))
        elif not self.server_running and not ContentProvider.is_port_available(self._current_port()):
            lines.append(f"   Warning: Can't start server, port {self._current_port()} is not available")

        if self.server_running:
            lines.append(make_selectable_text("Stop Server", True))

        lines.extend(self._get_tip_lines())

        return lines

    def _get_tip_lines(self) -> list[str]:
        tip_key = None
        if self.focus_idx == PORT_IDX:
            tip_key = "port"
        elif self.focus_idx == MAX_NODE_JOBS_IDX:
            tip_key = "max_node_jobs"
        elif self.focus_idx == MAX_API_JOBS_IDX:
            tip_key = "max_api_jobs"
        elif self.focus_idx == MAX_NODE_MEMORY_IDX:
            tip_key = "max_node_memory"
        elif self.focus_idx == MAX_CACHE_TIME_IDX:
            tip_key = "max_cache_time"
        elif self.focus_idx == API_KEYS_IDX:
            tip_key = "api_keys"

        if tip_key is not None:
            return ["", "Tip:", TIPS["jobs_server"][tip_key]]
        return []

    def get_footer(self) -> str:
        if self.server_running:
            return make_footer_text(["Enter: Stop Server", "Esc: Menu"])

        if self.focus_idx == PORT_IDX:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type Port", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == MAX_API_JOBS_IDX or self.focus_idx == MAX_CACHE_TIME_IDX or MAX_NODE_JOBS_IDX:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type", "Backspace: Remove character", "Esc: Menu"])

        if self.focus_idx == API_KEYS_IDX:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change", "Esc: Menu"])

        if self.focus_idx == SAVE_IDX:
            return make_footer_text(["Arrows U/D: Move", "Enter: Start Server", "Esc: Menu"])

        return ""

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running

    def can_start_server(self) -> bool:
        return self._valid_job_port() and \
            self._valid_max_api_jobs() and \
            self._valid_max_node_jobs() and \
            self._valid_max_node_memory() and \
            self._valid_max_cache_time() and \
            ContentProvider.is_port_available(self._current_port()) and \
            self._network_running() and \
            not self.server_running

    def _current_port(self) -> int | None:
        if self.edit_job_port == "":
            return DEFAULT_JOB_PORT
        try:
            return int(self.edit_job_port)
        except ValueError:
            return None
