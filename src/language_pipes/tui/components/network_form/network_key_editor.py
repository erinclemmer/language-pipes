

from enum import Enum
from typing import Callable, List, Optional

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.network_provider import NetworkProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.util.kb_utils import PressedKey

class NetworkKeyEditorState(Enum):
    LIST = 0
    INPUT = 1
    SHOW = 2

class NetworkKeyEditor:
    provider: ContentProvider
    confirm: Confirm
    exit_field_editor: Callable
    select_idx: int
    max_idx: int
    key: Optional[str]
    key_input: str
    key_valid: bool
    state: NetworkKeyEditorState

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_field_editor: Callable):
        self.provider = provider
        self.confirm = confirm
        self.exit_field_editor = exit_field_editor
        self.restart()

    def restart(self, reset_select: bool = True):
        if reset_select:
            self.select_idx = 0
        config = self.provider.network_provider.get_network_config()
        self.key = config.aes_key
        self.max_idx = 3 if self.key not in (None, "") else 1
        self.key_input = ""
        self.key_valid = False
        self.state = NetworkKeyEditorState.LIST

    def on_key(self, key: PressedKey, ch: str = ""):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Backspace:
            self.on_backspace()
        elif key == PressedKey.Alpha:
            self.on_char(ch)

    def on_next(self):
        self.select_idx = min(self.max_idx, self.select_idx + 1)

    def on_prev(self):
        self.select_idx = max(0, self.select_idx - 1)

    def validate_key(self):
        self.key_valid = NetworkProvider.validate_aes_key(self.key_input)

    def on_backspace(self):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input = self.key_input[:-1]
        self.validate_key()

    def on_char(self, ch: str):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input += ch
        self.validate_key()

    def on_enter(self):
        if self.select_idx == 0:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.INPUT
            elif self.state == NetworkKeyEditorState.INPUT:
                if self.key_valid:
                    self.confirm.open(
                        f"Save this network key?\n{self.key_input}",
                        on_apply=self.save_key,
                        on_discard=lambda: self.restart(False),
                        confirm_msg=f"Saved network key!\n{self.key_input}"
                    )
        elif self.select_idx == 1:
            self.confirm.open(
                "Generate a new network key?",
                on_apply=self.generate_key,
                on_discard=lambda: self.restart(False),
                confirm_msg="New network key generated"
            )
        elif self.select_idx == 2:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.SHOW
            elif self.state == NetworkKeyEditorState.SHOW:
                self.exit_field_editor()
        elif self.select_idx == 3:
            self.confirm.open(
                "Delete the current network key?\n\nThis will open the network to all connections unless a whitelist is set.",
                on_apply=self.delete_key,
                on_discard=lambda: self.restart(False),
                confirm_msg="Network key deleted"
            )

    def back(self) -> bool:
        self.restart(False)
        return self.state == NetworkKeyEditorState.LIST
    
    def save_key(self):
        config = self.provider.network_provider.get_network_config()
        config.aes_key = self.key_input
        self.provider.network_provider.save_network_config(config)
        self.exit_field_editor()

    def generate_key(self):
        config = self.provider.network_provider.get_network_config()
        config.aes_key = self.provider.network_provider.generate_aes_key()
        self.provider.network_provider.save_network_config(config)
        self.exit_field_editor()

    def delete_key(self):
        config = self.provider.network_provider.get_network_config()
        config.aes_key = None
        self.provider.network_provider.save_network_config(config)
        self.exit_field_editor()

    def get_list_lines(self) -> List[str]:
        lines = ["Editing Network Key"]

        def add_option(option: str, idx: int):
            l_cursor = "|>" if idx == self.select_idx else "  "
            r_cursor = "<|" if idx == self.select_idx else "  "
            lines.append(f" {l_cursor} {option} {r_cursor}")

        add_option("Enter Key", 0)
        add_option("Generate New Key", 1)
        if self.max_idx > 1:
            add_option("Show Existing Key", 2)
            add_option("Delete Existing Key", 3)
        lines.append("")
        return lines

    def get_input_lines(self) -> List[str]:
        key = (
            self.key_input[-40:]
            if len(self.key_input) > 40
            else self.key_input
        )
        lines = [
            "Enter AES hex key",
            "",
            f"Key: {key}",
            f"Length: {len(self.key_input)}",
            "",
        ]
        if self.key_valid:
            lines.append("Valid Key!")
        else:
            lines.append("Invalid key: must be a valid aes key hex string")
        return lines

    def get_show_lines(self) -> List[str]:
        key = self.key or ""
        return ["Network Key", "", key]

    def get_lines(self) -> List[str]:
        if self.state == NetworkKeyEditorState.LIST:
            return self.get_list_lines()
        if self.state == NetworkKeyEditorState.INPUT:
            return self.get_input_lines()
        if self.state == NetworkKeyEditorState.SHOW:
            return self.get_show_lines()
        return []

    def get_footer(self):
        if self.state == NetworkKeyEditorState.INPUT:
            return "[A-Z]: Type key   Backspace: delete char   Esc: Back   Enter: Accept"
        if self.state == NetworkKeyEditorState.SHOW:
            return "Enter/Esc: Back"
        return "Arrows U/D: Change choice   Enter: Confirm   Esc: Back"