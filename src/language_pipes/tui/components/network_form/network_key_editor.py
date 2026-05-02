

from enum import Enum
from typing import Callable, Optional

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.network_provider import NetworkProvider
from language_pipes.tui.components.confirm import Confirm

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

    def get_footer(self):
        if self.state == NetworkKeyEditorState.INPUT:
            return "[A-Z]: Type key   Backspace: delete char   Esc: Back   Enter: Accept"
        if self.state == NetworkKeyEditorState.SHOW:
            return "Enter/Esc: Back"
        return "Arrows U/D: Change choice   Enter: Confirm   Esc: Back"