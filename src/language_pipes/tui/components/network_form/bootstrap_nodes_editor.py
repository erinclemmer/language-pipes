from typing import Callable, List

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint
from language_pipes.tui.components.network_form.util import validate_address, validate_port
from language_pipes.tui.components.network_form.list_editor import ListEditor


class BootstrapNodesEditor(ListEditor[Endpoint]):
    new_node_address: str
    new_node_port: str
    new_address_valid: bool
    new_port_valid: bool

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_editor: Callable):
        super().__init__(provider, confirm, exit_editor)

    # ------------------------------------------------------------------
    # ListEditor abstract implementation
    # ------------------------------------------------------------------

    def load_items(self) -> List[Endpoint]:
        config = self.provider.network_provider.get_network_config()
        return config.bootstrap_nodes

    def reset_input_fields(self) -> None:
        self.new_node_address = ""
        self.new_node_port = "5000"
        self.new_address_valid = False
        self.new_port_valid = True
        self._validate_new_node()

    def is_input_valid(self) -> bool:
        return self.new_address_valid and self.new_port_valid

    def input_field_count(self) -> int:
        return 2

    def on_save_new(self, discard: Callable) -> None:
        def save_node():
            config = self.provider.network_provider.get_network_config()
            config.bootstrap_nodes.append(Endpoint(self.new_node_address, int(self.new_node_port)))
            self.provider.network_provider.save_network_config(config)
            self.restart()

        self.confirm.open(
            f"Add \"{self.new_node_address}:{self.new_node_port}\"?",
            save_node,
            discard,
            f"Added {self.new_node_address}:{self.new_node_port} to bootstrap nodes",
        )

    def on_select_existing(self, item: Endpoint, discard: Callable) -> None:
        # No action for selecting an existing bootstrap node
        pass

    def on_delete_existing(self, item: Endpoint) -> None:
        def on_apply():
            config = self.provider.network_provider.get_network_config()
            config.bootstrap_nodes = [n for i, n in enumerate(self.items) if i != self.select_idx]
            self.provider.network_provider.save_network_config(config)
            self.restart()

        def on_discard():
            pass

        self.confirm.open(
            f"Delete \"{item.address}:{item.port}\"?",
            on_apply,
            on_discard,
            f"Removed {item.address}:{item.port}",
        )

    def format_item(self, item: Endpoint) -> str:
        return f"{item.address}:{item.port}"

    def get_input_lines(self) -> List[str]:
        ip_cursor = "|" if self.focus_idx == 0 else ""
        port_cursor = "|" if self.focus_idx == 1 else ""
        return [
            "Type the port and address of a new node",
            "",
            f"IP Address|> {self.new_node_address}{ip_cursor}",
            f" Peer Port|> {self.new_node_port}{port_cursor}",
            "", "",
            "Invalid Address" if not self.new_address_valid else "Valid Address",
            "Invalid Port" if not self.new_port_valid else "Valid Port",
        ]

    def input_footer(self) -> str:
        return "[A-Z]: Type   Backspace: delete char   Esc: Discard   Enter: Accept"

    def list_footer(self) -> str:
        return "Arrows U/D: Change choice   Enter: Select   Esc: Back   Delete: Unregister"

    def list_header(self) -> str:
        return "Bootstrap Nodes:"

    def add_new_label(self) -> str:
        return "Add New Node"

    def on_char_to_field(self, ch: str) -> None:
        if self.focus_idx == 0:
            self.new_node_address += ch
        else:
            self.new_node_port += ch
        self._validate_new_node()

    def on_backspace_field(self) -> None:
        if self.focus_idx == 0:
            self.new_node_address = self.new_node_address[:-1]
        else:
            self.new_node_port = self.new_node_port[:-1]
        self._validate_new_node()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_new_node(self) -> None:
        self.new_address_valid = validate_address(self.new_node_address)
        self.new_port_valid = validate_port(self.new_node_port)
