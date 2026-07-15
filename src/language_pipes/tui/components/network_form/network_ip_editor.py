from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from ansinout import PressedKey
from language_pipes.tui.components.confirm import Confirm

from distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.components.network_form.util import validate_address


class NetworkIpEditor:
    confirm: Confirm
    provider: ContentProvider
    exit_editor: Callable

    network_ip: str
    valid_address: bool

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_editor: Callable):
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.provider = provider
        self.restart()

    def restart(self):
        config = self.provider.network_provider.get_network_config()
        self.network_ip = config.network_ip if config.network_ip is not None else ""
        self.validate_address()

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Backspace:
            self.on_backspace()
        elif key in (PressedKey.Alpha, PressedKey.Paste):
            self.on_char(ch)

    def back(self) -> bool:
        return True

    def on_enter(self):
        if self.valid_address:
            self.confirm.open(
                f"Set {self.network_ip} as node IP address?",
                on_apply=self.save_address,
                confirm_msg=f"Saved {self.network_ip} as node IP address",
                on_discard=self.restart
            )

    def save_address(self):
        config: DSNodeConfig = self.provider.network_provider.get_network_config()
        config.network_ip = self.network_ip
        self.provider.network_provider.save_network_config(config)
        self.exit_editor()

    def on_char(self, ch: str):
        self.network_ip += ch
        self.validate_address()

    def validate_address(self):
        self.valid_address = validate_address(self.network_ip)

    def on_backspace(self):
        self.network_ip = self.network_ip[:-1]
        self.validate_address()

    def get_footer(self):
        return "[A-Z]: Type IP Address   Backspace: delete char   Esc: Discard   Enter: Accept"

    def get_lines(self):
        lines = [
            "Type the IP address other nodes can reach this node on",
            "",
            f"IP Address|> {self.network_ip}|"
        ]

        if self.valid_address:
            lines.extend([
                "", "Valid IP!"
            ])
        else:
            lines.extend([
                "", "Invalid IP Address"
            ])
        
        return lines
