from typing import Callable, List

from language_pipes.tui.components.confirm import Confirm
from language_pipes.content_loader import ContentLoader, ProviderCall
from language_pipes.tui.components.network_form.list_editor import ListEditor
from language_pipes.distributed_state_network.objects.config import DSNodeConfig

class WhitelistEditor(ListEditor[str]):
    new_node_id: str

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        super().__init__(loader, confirm, exit_editor)

    # ------------------------------------------------------------------
    # ListEditor abstract implementation
    # ------------------------------------------------------------------

    def load_items(self) -> List[str]:
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        return config.whitelist_node_ids

    def reset_input_fields(self) -> None:
        self.new_node_id = ""

    def is_input_valid(self) -> bool:
        return True

    def input_field_count(self) -> int:
        return 1

    def on_save_new(self, discard: Callable) -> None:
        def save_node():
            config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
            config.whitelist_node_ids.append(self.new_node_id)
            self.loader.call_provider(ProviderCall.save_network_config, config)
            self.restart()

        self.confirm.open(
            f"Add \"{self.new_node_id}\"?",
            save_node,
            discard,
            f"Added {self.new_node_id} to white list",
        )

    def on_select_existing(self, item: str, discard: Callable) -> None:
        # No action for selecting an existing bootstrap node
        pass

    def on_delete_existing(self, item: str) -> None:
        def on_apply():
            config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
            config.whitelist_node_ids = [n for i, n in enumerate(self.items) if i != self.select_idx]
            self.loader.call_provider(ProviderCall.save_network_config, config)
            self.restart()

        def on_discard():
            pass

        self.confirm.open(
            f"Delete \"{item}\"?",
            on_apply,
            on_discard,
            f"Removed {item}",
        )

    def format_item(self, item: str) -> str:
        return item

    def get_input_lines(self) -> List[str]:
        return [
            "Type the ID of the node to add to the whitelist",
            "",
            f"Node ID|> {self.new_node_id}"
        ]

    def input_footer(self) -> str:
        return "[A-Z]: Type   Backspace: delete char   Esc: Discard   Enter: Accept"

    def list_footer(self) -> str:
        return "Arrows U/D: Change choice   Enter: Select   Esc: Back   Delete: Remove"

    def list_header(self) -> str:
        return "Bootstrap Nodes:"

    def add_new_label(self) -> str:
        return "Add New Node"

    def on_char_to_field(self, ch: str) -> None:
        self.new_node_id += ch

    def on_backspace_field(self) -> None:
        self.new_node_id = self.new_node_id[:-1]
