from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm

from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.components.network_form.util import validate_port

class PeerPortEditor:
    confirm: Confirm
    provider: ContentProvider
    exit_editor: Callable

    peer_port_str: str
    valid_port: bool

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_editor: Callable):
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.provider = provider
        self.restart()

    def restart(self):
        config: DSNodeConfig = self.provider.network_provider.get_network_config()
        self.peer_port_str = str(config.port)
        self.validate_port()

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Backspace:
            self.on_backspace()
        elif key == PressedKey.Alpha:
            self.on_char(ch)

    def back(self) -> bool:
        return True

    def on_enter(self):
        if self.valid_port:
            self.confirm.open(
                f"Set {self.peer_port_str} as peer port?",
                on_apply=self.save_port,
                confirm_msg=f"Saved {self.peer_port_str} as peer port",
                on_discard=self.restart
            )

    def save_port(self):
        config: DSNodeConfig = self.provider.network_provider.get_network_config()
        config.port = int(self.peer_port_str)
        self.provider.network_provider.save_network_config(config)
        self.exit_editor()

    def on_char(self, ch: str):
        self.peer_port_str += ch
        self.validate_port()

    def validate_port(self):
        self.valid_port = validate_port(self.peer_port_str)

    def on_backspace(self):
        self.peer_port_str = self.peer_port_str[:-1]
        self.validate_port()

    def get_footer(self):
        return "[A-Z]: Type Port   Backspace: delete char   Esc: Discard   Enter: Accept"

    def get_lines(self):
        lines = [
            "Type an available TCP port",
            "",
            f"Peer Port|> {self.peer_port_str}|"
        ]

        if self.valid_port:
            lines.extend([
                "", "Valid Port"
            ])
        else:
            lines.extend([
                "", "Invalid Port"
            ])
        
        return lines
