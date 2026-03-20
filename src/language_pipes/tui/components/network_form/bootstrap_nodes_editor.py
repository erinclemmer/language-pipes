from typing import Callable, List

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint

class BootstrapNodesEditor:
    confirm: Confirm
    loader: ContentLoader
    exit_editor: Callable

    bootstrap_nodes: List[Endpoint]

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.loader = loader
        self.restart()

    def restart(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        self.bootstrap_nodes = config.bootstrap_nodes

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
        pass

    def save(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.bootstrap_nodes = self.bootstrap_nodes
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

    def on_char(self, ch: str):
        pass

    def on_backspace(self):
        pass

    def get_footer(self):
        return "[A-Z]: Type IP Address   Backspace: delete char   Esc: Discard   Enter: Accept"

    def get_lines(self):
        lines = [
            
        ]

        
        return lines
