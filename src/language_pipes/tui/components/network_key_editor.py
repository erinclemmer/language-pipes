from enum import Enum
from typing import Callable, List

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig

class NetworkKeyEditorState(Enum):
    LIST = 0
    INPUT = 1
    GENERATE = 2
    SHOW = 3
    DELETE = 4
    
class NetworkKeyEditor:
    confirm: Confirm
    loader: ContentLoader
    exit_editor: Callable

    key_input: str
    max_idx: int
    select_idx: int
    state: NetworkKeyEditorState

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.loader = loader
        self.confirm = confirm
        self.exit_editor = exit_editor
        self.restart()

    def restart(self):
        self.select_idx = 0
        self.state = NetworkKeyEditorState.LIST
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        self.network_key = config.aes_key
        self.key_input = ""
        self.key_valid = False
        self.max_idx = 3 if self.network_key is not None and self.network_key != '' else 1

    def get_footer(self):
        return "Arrows U/D: Change choice   Enter: Confirm   Esc: Back"

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Alpha:
            self.on_char(ch)
        elif key == PressedKey.Backspace:
            self.on_backspace()

    def on_backspace(self):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input = self.key_input[:-1]

    def on_char(self, ch: str):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input += ch
        self.key_valid = self.loader.call_provider(ProviderCall.validate_aes_key, self.key_input)

    def on_prev(self):
        self.select_idx -= 1
        if self.select_idx < 0:
            self.select_idx = 0
    
    def on_next(self):
        self.select_idx += 1
        if self.select_idx > self.max_idx:
            self.select_idx = self.max_idx

    def on_enter(self):
        if self.select_idx == 0:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.INPUT
            elif self.state == NetworkKeyEditorState.INPUT:
                if self.key_valid:
                    self.confirm.open(
                        f"Save this network key?\n{self.key_input[:32]}\n{self.key_input[32:]}",
                        on_apply=self.save_key_input,
                        on_discard=self.exit_editor,
                        confirm_msg=f"Saved network key!\n{self.key_input[:32]}\n{self.key_input[32:]}"
                    )
        elif self.select_idx == 1:
            self.confirm.open(
                "Generate a new network key?",
                on_apply=self.generate_key,
                on_discard=self.exit_editor,
                confirm_msg="New network key generated"
            )
        elif self.select_idx == 2:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.SHOW
            elif self.state == NetworkKeyEditorState.SHOW:
                self.exit_editor()
        elif self.select_idx == 3:
            self.confirm.open(
                "Delete the current network key?\n\nThis will open the network to all\nconnections unless a whitelist is set.",
                on_apply=self.delete_key,
                confirm_msg="Network key deleted",
                on_discard=self.exit_editor
            )

    def generate_key(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.aes_key = self.loader.call_provider(ProviderCall.generate_aes_key)
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

    def save_key_input(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.aes_key = self.key_input
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

    def delete_key(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.aes_key = None
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

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
        key = self.key_input[-40:] if len(self.key_input) > 40 else self.key_input
        lines = [
            "Enter AES hex key",
            "",
            f"Key: {key}",
            f"Length: {len(self.key_input)}",
            ""
        ]

        if self.key_valid:
            lines.append("Valid Key!")
        else:
            lines.append("Invalid key: must be a valid aes key hex string")

        return lines

    def get_show_lines(self) -> List[str]:
        key = ""
        if self.network_key is not None:
            key = self.network_key
        lines = [
            "Network Key",
            "",
            key[:32],
            key[32:]
        ]

        return lines

    def get_lines(self) -> List[str]:
        if self.state == NetworkKeyEditorState.LIST:
            return self.get_list_lines()
        if self.state == NetworkKeyEditorState.INPUT:
            return self.get_input_lines()
        if self.state == NetworkKeyEditorState.SHOW:
            return self.get_show_lines()
        return []
