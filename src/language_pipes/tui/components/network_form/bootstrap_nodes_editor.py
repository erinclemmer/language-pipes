from typing import Callable, List

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint
from language_pipes.tui.components.network_form.util import validate_address, validate_port

class BootstrapNodesEditor:
    confirm: Confirm
    loader: ContentLoader
    exit_editor: Callable

    select_idx: int
    focus_idx: int
    bootstrap_nodes: List[Endpoint]

    adding_bootstrap_node: bool
    new_node_address: str
    new_node_port: str
    new_address_valid: bool
    new_port_valid: bool

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.loader = loader
        self.restart()

    def restart(self, reset_select: bool = True):
        if reset_select:
            self.select_idx = 0
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        self.bootstrap_nodes = config.bootstrap_nodes
        self.adding_bootstrap_node = True if len(self.bootstrap_nodes) == 0 else False
        self.new_node_address = ""
        self.new_node_port = "5000"
        self.focus_idx = 0
        self.validate_new_node()

    def validate_new_node(self):
        self.new_address_valid = validate_address(self.new_node_address)
        self.new_port_valid = validate_port(self.new_node_port)

    def on_key(self, key: PressedKey, ch: str):
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
        elif key == PressedKey.Delete:
            self.on_delete()

    def on_next(self):
        if self.adding_bootstrap_node:
            self.focus_idx += 1
            if self.focus_idx > 1:
                self.focus_idx = 0
        else:
            self.select_idx += 1
            if self.select_idx > len(self.bootstrap_nodes):
                self.select_idx = 0

    def on_prev(self):
        if self.adding_bootstrap_node:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                self.focus_idx = 1
        else:
            self.select_idx -= 1
            if self.select_idx < 0:
                self.select_idx = len(self.bootstrap_nodes)

    def back(self) -> bool:
        adding = self.adding_bootstrap_node
        if adding:
            self.restart(False)
        return not adding or len(self.bootstrap_nodes) == 0

    def on_enter(self):
        def discard_choice():
            self.restart()
            self.confirm.close()

        if self.adding_bootstrap_node:
            if not self.new_port_valid or not self.new_address_valid:
                return
            
            def save_node_id():
                config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
                config.bootstrap_nodes.append(Endpoint(self.new_node_address, int(self.new_node_port)))
                self.loader.call_provider(ProviderCall.save_network_config, config)
                self.restart()
            self.confirm.open(f"Add \"{self.new_node_address}:{self.new_node_port}\"?", save_node_id, discard_choice, f"Added {self.new_node_address}:{self.new_node_port} to bootstrap nodes")
            return

        if self.select_idx == len(self.bootstrap_nodes):
            self.adding_bootstrap_node = True

    def on_char(self, ch: str):
        if not self.adding_bootstrap_node:
            return
        if self.focus_idx == 0:
            self.new_node_address += ch
        else:
            self.new_node_port += ch

        self.validate_new_node()

    def on_backspace(self):
        if not self.adding_bootstrap_node:
            return
        if self.focus_idx == 0:
            self.new_node_address = self.new_node_address[:-1]
        else:
            self.new_node_port = self.new_node_port[:-1]
        self.validate_new_node()

    def on_delete(self):
        selected_node = self.bootstrap_nodes[self.select_idx]
        def on_apply():
            config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
            config.bootstrap_nodes = [n for i, n in enumerate(self.bootstrap_nodes) if i != self.select_idx]
            self.loader.call_provider(ProviderCall.save_network_config, config)
            self.restart()

        def on_discard():
            pass
        self.confirm.open(
            f"Delete \"{selected_node.address}:{selected_node.port}\"?", 
            on_apply, on_discard,
            f"Removed {selected_node.address}:{selected_node.port}"
        )

    def get_footer(self):
        if self.adding_bootstrap_node:
            return "[A-Z]: Type   Backspace: delete char   Esc: Discard   Enter: Accept"
        else:
            return "Arrows U/D: Change choice   Enter: Select   Esc: Discard   Delete: Unregister"

    def get_lines(self):
        lines = []

        if self.adding_bootstrap_node:
            ip_cursor = "|" if self.focus_idx == 0 else ""
            port_cursor = "|" if self.focus_idx == 1 else ""
            lines.extend([
                "Type the port and address of a new node",
                "",
                f"IP Address|> {self.new_node_address}{ip_cursor}",
                f" Peer Port|> {self.new_node_port}{port_cursor}",
                "", "",
                "Invalid Address" if not self.new_address_valid else "Valid Address",
                "Invalid Port" if not self.new_port_valid else "Valid Port"
            ])
        else:
            if len(self.bootstrap_nodes) > 0:
                lines.append("Bootstrap Nodes:")
                for i, node in enumerate(self.bootstrap_nodes):
                    l_cursor = "|>" if i == self.select_idx else "  "
                    r_cursor = "<|" if i == self.select_idx else "  "
                    lines.append(f" {l_cursor} {node.address}:{node.port} {r_cursor}")
                lines.append("")
            l_cursor = "|>" if self.select_idx == len(self.bootstrap_nodes) else "  "
            r_cursor = "<|" if self.select_idx == len(self.bootstrap_nodes) else "  "
            lines.append(f" {l_cursor} Add New Node {r_cursor}")

        return lines
