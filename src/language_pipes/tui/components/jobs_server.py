import secrets
from enum import Enum
import time
from typing import Callable, List, Tuple

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from ansinout import PressedKey
from language_pipes.tui.util.text import make_footer_text, make_selectable_text

class JobsServerState(Enum):
    Top = 'top'
    Keys = 'keys'
    AddKeyType = 'add_key_type'
    KeyGen = 'key_gen'
    TypeKey = 'type_key'

class JobsServer:
    confirm: Confirm
    provider: ContentProvider
    exit_page: Callable

    state: JobsServerState
    focus_idx: int
    key_idx: int
    type_idx: int
    server_running: bool
    oai_port: int
    api_keys: List[str]

    edit_oai_port: str
    new_api_key: str

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_page: Callable):
        self.provider = provider
        self.confirm = confirm
        self.exit_page = exit_page
        self.server_running = False
        self.focus_idx = 0
        self.key_idx = 0
        self.type_idx = 0
        self.state = JobsServerState.Top
        self.oai_port = self.provider.job_provider.get_oai_port()
        self.api_keys = self.provider.job_provider.get_api_keys()
        self.edit_oai_port = str(self.oai_port)

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Escape:
            self.on_escape()
        elif key == PressedKey.Alpha:
            self.on_char(ch)
        elif key == PressedKey.Backspace:
            self.on_backspace()
        elif key == PressedKey.Delete:
            self.on_delete()
    
    def on_delete(self):
        if self.state == JobsServerState.Keys and self.key_idx < len(self.api_keys):
            def on_apply():
                self.api_keys = [key for key in self.api_keys if key != self.api_keys[self.key_idx]]
                self.provider.job_provider.set_api_keys(self.api_keys)

            self.confirm.open(
                f"Delete {self.api_keys[self.key_idx]}?",
                on_apply=on_apply,
                on_discard=lambda:None
            )

    def on_char(self, ch: str):
        if self.state == JobsServerState.Top and self.focus_idx == 0:
            self.edit_oai_port += ch
        elif self.state == JobsServerState.TypeKey:
            self.new_api_key += ch

    def on_backspace(self):
        if self.state == JobsServerState.Top and self.focus_idx == 0:
            self.edit_oai_port = self.edit_oai_port[:-1]
        if self.state == JobsServerState.TypeKey:
            self.new_api_key = self.new_api_key[:-1]

    def on_escape(self):
        if self.state == JobsServerState.Top:
            self.exit_page()
        elif self.state == JobsServerState.Keys:
            self.state = JobsServerState.Top
        elif self.state == JobsServerState.AddKeyType:
            self.state = JobsServerState.Keys
        elif self.state == JobsServerState.KeyGen:
            self.state = JobsServerState.Keys
        elif self.state == JobsServerState.TypeKey:
            self.state = JobsServerState.Keys

    def on_enter(self):
        if self.state == JobsServerState.Top:
            if self.focus_idx == 0:
                self.focus_idx = 1
            elif self.focus_idx == 1:
                self.state = JobsServerState.Keys
            elif self.focus_idx == 2:
                if self.server_running:
                    self.provider.job_provider.stop_oai_server()
                else:
                    self.save_and_run()
        elif self.state == JobsServerState.Keys:
            if self.key_idx == len(self.api_keys):
                self.state = JobsServerState.AddKeyType
        elif self.state == JobsServerState.AddKeyType:
            if self.type_idx == 0:
                self.state = JobsServerState.KeyGen
                self.new_api_key = secrets.token_urlsafe(32)
            if self.type_idx == 1:
                self.state = JobsServerState.TypeKey
                self.new_api_key = ""
            if self.type_idx == 2:
                self.state = JobsServerState.Keys
        elif self.state == JobsServerState.KeyGen:
            self.api_keys.append(self.new_api_key)
            self.provider.job_provider.set_api_keys(self.api_keys)
            self.state = JobsServerState.Keys
        elif self.state == JobsServerState.TypeKey:
            if len(self.new_api_key) > 0:
                self.api_keys.append(self.new_api_key)
                self.provider.job_provider.set_api_keys(self.api_keys)
                self.state = JobsServerState.Keys
    
    def save_and_run(self):
        if not self.validate_oai_port():
            return
        
        self.provider.job_provider.set_oai_port(int(self.edit_oai_port))
        self.provider.job_provider.set_api_keys(self.api_keys)
        self.provider.job_provider.start_oai_server()

    def on_prev(self):
        if self.state == JobsServerState.Top and not self.server_running:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                max_idx = 2 if self.can_start_server() else 1
                self.focus_idx = max_idx
        elif self.state == JobsServerState.Keys:
            self.key_idx -= 1
            if self.key_idx < 0:
                self.key_idx = len(self.api_keys)
        elif self.state == JobsServerState.AddKeyType:
            self.type_idx -= 1
            if self.type_idx < 0:
                self.type_idx = 2

    def on_next(self):
        if self.state == JobsServerState.Top and not self.server_running:
            self.focus_idx += 1
            max_idx = 2 if self.can_start_server() else 1
            if self.focus_idx > max_idx:
                self.focus_idx = 0
        elif self.state == JobsServerState.Keys:
            self.key_idx += 1
            if self.key_idx > len(self.api_keys):
                self.key_idx = 0
        elif self.state == JobsServerState.AddKeyType:
            self.type_idx += 1
            if self.type_idx > 2:
                self.type_idx = 0

    def get_view(self):
        if self.state == JobsServerState.Top:
            return self.get_top_view()
        elif self.state == JobsServerState.Keys:
            return self.get_keys_view()
        elif self.state == JobsServerState.AddKeyType:
            return self.get_ask_key_type_view()
        elif self.state == JobsServerState.KeyGen:
            return self.get_gen_key_view()
        elif self.state == JobsServerState.TypeKey:
            return self.get_type_key_view()

    def get_type_key_view(self) -> List[str]:
        lines = [
            "Type New API Key:", "",
            f"New Key: {self.new_api_key}|", "",
        ]

        if len(self.new_api_key) == 0:
            lines.append("WARNING: Key is empty, cannot save")

        lines.append("Press Enter to Accept key or Escape to discard key")

        return lines

    def get_gen_key_view(self) -> List[str]:
        lines = [
            "Generate Key:", "",
            f"New Key: {self.new_api_key}", "",
            "Press Enter to Accept or Escape to go back" 
        ]
        return lines

    def get_ask_key_type_view(self) -> List[str]:
        lines = ["Add New Key:", ""]
        for i, opt in enumerate(['Generate', 'Enter Manually', 'Back']):
            lines.append(make_selectable_text(opt, self.type_idx == i))

        return lines

    def get_keys_view(self) -> List[str]:
        lines = ["API Keys:", ""]

        for i, key in enumerate(self.api_keys):
            lines.append(make_selectable_text(key, self.key_idx == i))
        
        lines.append("")
        lines.append(make_selectable_text("Add new key", self.key_idx == len(self.api_keys)))

        return lines

    def get_top_view(self) -> List[str]:
        port_cursor = "|" if self.focus_idx == 0 else ""
        self.server_running = self.provider.job_provider.oai_server_running()
        port_string = f"   Port: {self.edit_oai_port}{port_cursor}" if not self.server_running else f"   Running Server on port {self.edit_oai_port}"
        lines = [
            "Jobs Server:", "",
            port_string
        ]
        
        if not self.validate_oai_port():
            lines.append("   Error: Invalid port value")
        
        lines.append(make_selectable_text(f"{len(self.api_keys)} api key(s)", self.focus_idx == 1))
        if len(self.api_keys) == 0:
            lines.extend(["   INFO: No API keys set, authentication not required", ""])

        if not self._network_running():
            lines.append("   WARNING: Network must be connected before server can start")

        if self.server_running:
            lines.extend(["   INFO: Stop server to edit port and API Keys", ""])
        
        if self.can_start_server():
            lines.append(make_selectable_text("Start Server", self.focus_idx == 2))
        elif not self.server_running and not ContentProvider.is_port_available(self.oai_port):
            lines.append(f"   Warning: Can't start server, port {self.oai_port} is not available")

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
    
    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running

    def can_start_server(self) -> bool:
        return ContentProvider.is_port_available(self.oai_port) and self.validate_oai_port() and self._network_running()
    
    def validate_oai_port(self) -> bool:
        try:
            port = int(self.edit_oai_port)
            return port < 65000 and port > 0
        except ValueError:
            return False

    def get_footer(self) -> str:
        if self.state == JobsServerState.Top and self.focus_idx == 0 and not self.server_running:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type Port", "Backspace: Remove character", "Esc: Menu"])
        
        if self.state == JobsServerState.Top and self.focus_idx == 1 and not self.server_running:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change API keys", "Esc: Menu"])
        
        if self.state == JobsServerState.Top and self.focus_idx == 2 and not self.server_running:
            return make_footer_text(["Arrows U/D: Move", "Enter: Start Server", "Esc: Menu"])
        
        if self.state == JobsServerState.Top and self.server_running:
            return make_footer_text(["Enter: Stop Server", "Esc: Menu"])

        if self.state == JobsServerState.Keys and self.key_idx == len(self.api_keys):
            return make_footer_text(["Arrows U/D: Move", "Enter: Add new key", "Esc: Back"])
        
        if self.state == JobsServerState.Keys and self.key_idx < len(self.api_keys):
            return make_footer_text(["Arrows U/D: Move", "Delete: Remove API key", "Esc: Back"])

        if self.state == JobsServerState.AddKeyType:
            return make_footer_text(["Arrows U/D: Move", "Enter: Select option", "Esc: Back"])

        if self.state == JobsServerState.KeyGen:
            return make_footer_text(["Enter: Accept key", "Escape: Discard key"])

        return ""