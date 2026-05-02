from typing import Callable, List

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm

from language_pipes.tui.components.network_form.list_editor import ListEditor
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.util.text import make_footer_text

class WhitelistEditor(ListEditor[str]):
    new_node_id: str

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_editor: Callable):
        super().__init__(provider, confirm, exit_editor)

    def load_items(self) -> List[str]:
        config: DSNodeConfig = self.provider.network_provider.get_network_config()
        return config.whitelist_node_ids

    def reset_input_fields(self) -> None:
        self.new_node_id = ""

    def is_input_valid(self) -> bool:
        return len(self.new_node_id) > 0

    def input_field_count(self) -> int:
        return 1

    def on_save_new(self, discard: Callable) -> None:
        def save_node():
            config: DSNodeConfig = self.provider.network_provider.get_network_config()
            config.whitelist_node_ids.append(self.new_node_id)
            self.provider.network_provider.save_network_config(config)
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
            config: DSNodeConfig = self.provider.network_provider.get_network_config()
            config.whitelist_node_ids = [n for i, n in enumerate(self.items) if i != self.select_idx]
            self.provider.network_provider.save_network_config(config)
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
        return make_footer_text(["[A-Z]: Type", "Backspace: remove character", "Enter: Accept", "Esc: Discard"])

    def list_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Change choice", "Enter: Select", "Delete: Remove", "Esc: Back"])

    def list_header(self) -> str:
        return "Bootstrap Nodes:"

    def add_new_label(self) -> str:
        return "Add New Node"

    def on_char_to_field(self, ch: str) -> None:
        self.new_node_id += ch

    def on_backspace_field(self) -> None:
        self.new_node_id = self.new_node_id[:-1]
