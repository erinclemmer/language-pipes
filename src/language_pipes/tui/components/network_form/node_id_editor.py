from typing import Callable, List

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm

from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.components.network_form.list_editor import ListEditor


class NodeIdEditor(ListEditor[str]):
    new_node_id: str

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_editor: Callable):
        self.selected_node_id = None
        super().__init__(provider, confirm, exit_editor)

    # ------------------------------------------------------------------
    # Expose registering_node_id as an alias for the base adding flag
    # so existing code (e.g. tests) that reads editor.registering_node_id
    # continues to work.
    # ------------------------------------------------------------------

    @property
    def registering_node_id(self) -> bool:
        return self.adding

    @registering_node_id.setter
    def registering_node_id(self, value: bool) -> None:
        self.adding = value

    # Keep node_ids as an alias for items for the same reason.
    @property
    def node_ids(self) -> List[str]:
        return self.items

    @node_ids.setter
    def node_ids(self, value: List[str]) -> None:
        self.items = value

    # ------------------------------------------------------------------
    # ListEditor abstract implementation
    # ------------------------------------------------------------------

    def load_items(self) -> List[str]:
        return self.provider.network_provider.get_registered_node_ids()

    def reset_input_fields(self) -> None:
        self.new_node_id = ""

    def is_input_valid(self) -> bool:
        return self.new_node_id != ""

    def input_field_count(self) -> int:
        return 1

    def on_save_new(self, discard: Callable) -> None:
        def save_node_id():
            config: DSNodeConfig = self.provider.network_provider.get_network_config()
            config.node_id = self.new_node_id
            self.provider.network_provider.save_new_node_id(self.new_node_id)
            self.provider.network_provider.save_network_config(config)
            self.confirm.close()
            self.exit_editor()

        self.confirm.open(
            f"Register \"{self.new_node_id}\"?",
            save_node_id,
            discard,
            f"Registered new keys for {self.new_node_id}\nand set to current node ID",
        )

    def on_select_existing(self, item: str, discard: Callable) -> None:
        selected_node_id = item

        def use_node_id():
            config: DSNodeConfig = self.provider.network_provider.get_network_config()
            config.node_id = selected_node_id
            self.provider.network_provider.save_network_config(config)
            self.confirm.close()
            self.exit_editor()

        self.confirm.open(
            f"Use \"{selected_node_id}\"?",
            use_node_id,
            discard,
            f"Set node ID to {selected_node_id}",
        )

    def on_delete_existing(self, item: str) -> None:
        def on_apply():
            self.provider.network_provider.delete_node_id(item)
            self.restart()

        def on_discard():
            pass

        self.confirm.open(
            f"Unregister \"{item}\"",
            on_apply,
            on_discard,
            f"Deleted keys for {item}",
        )

    def format_item(self, item: str) -> str:
        return item

    def get_input_lines(self) -> List[str]:
        return [
            "Type a new node ID then press Enter to save",
            "",
            f"New Node ID|> {self.new_node_id}|",
        ]

    def input_footer(self) -> str:
        return "[A-Z]: Type node ID   Backspace: delete char   Esc: Discard   Enter: Accept"

    def list_footer(self) -> str:
        return "Arrows U/D: Change choice   Enter: Select   Esc: Discard   Delete: Unregister"

    def list_header(self) -> str:
        return "Registered Node IDs:"

    def add_new_label(self) -> str:
        return "Register new node id"

    def on_char_to_field(self, ch: str) -> None:
        self.new_node_id += ch

    def on_backspace_field(self) -> None:
        self.new_node_id = self.new_node_id[:-1]

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        """Override to handle the empty-input discard case and to load
        node_ids from the provider before the base logic runs."""
        def discard_choice():
            self.restart()
            self.confirm.close()

        if self.adding:
            if self.new_node_id == "":
                discard_choice()
                return
            self.on_save_new(discard_choice)
            return

        if self.select_idx == len(self.items):
            self.adding = True
        else:
            self.on_select_existing(self.items[self.select_idx], discard_choice)