from typing import Callable, List

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint

class NodeIdEditor:
    node_ids: List[str]
    confirm: Confirm
    loader: ContentLoader
    select_idx: int
    exit_editor: Callable

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.select_idx = 0
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.loader = loader
        self.node_ids = []
        self.registering_node_id = False
        self.new_node_id = ""
        self.selected_node_id = None

    def restart(self):
        self.new_node_id = ""
        self.node_ids = []
        self.select_idx = 0
        self.registering_node_id = False

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
        self.select_idx += 1
        if self.select_idx > len(self.node_ids):
            self.select_idx = 0

    def on_prev(self):
        self.select_idx -= 1
        if self.select_idx < 0:
            self.select_idx = len(self.node_ids)

    def on_enter(self):
        def discard_choice():
            self.restart()
            self.confirm.close()
            self.exit_editor()

        if self.registering_node_id:
            if self.new_node_id == "":
                discard_choice()
                return
            
            def save_node_id():
                config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
                config.node_id = self.new_node_id
                self.loader.call_provider(ProviderCall.save_new_node_id, self.new_node_id)
                self.loader.call_provider(ProviderCall.save_network_config, config)
                self.confirm.close()
                self.exit_editor()
            self.confirm.open(f"Register \"{self.new_node_id}\"?", save_node_id, discard_choice, f"Registered new keys for {self.new_node_id}\nand set to current node ID")
            return

        if self.select_idx == len(self.node_ids):
            self.registering_node_id = True
        else:
            selected_node_id = self.node_ids[self.select_idx]
            def use_node_id():
                config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
                config.node_id = selected_node_id
                self.loader.call_provider(ProviderCall.save_network_config, config)
                self.confirm.close()
                self.exit_editor()
            self.confirm.open(f"Use \"{selected_node_id}\"?", use_node_id, discard_choice, f"Set node ID to {selected_node_id}")

    def on_char(self, ch: str):
        self.new_node_id += ch

    def on_backspace(self):
        self.new_node_id = self.new_node_id[:-1]

    def on_delete(self):
        selected_node_id = self.node_ids[self.select_idx]
        def on_apply():
            self.loader.call_provider(ProviderCall.delete_node_id, selected_node_id)
            self.restart()

        def on_discard():
            pass
        self.confirm.open(f"Unregister \"{selected_node_id}\"", on_apply, on_discard, f"Deleted keys for {selected_node_id}")

    def get_footer(self):
        if self.registering_node_id:
            return "[A-Z]: Type node ID   Backspace: delete char   Esc: Discard   Enter: Accept"
        else:
            return "Arrows U/D: Change choice   Enter: Select   Esc: Discard   Delete: Unregister"

    def get_lines(self):
        lines = []

        if self.registering_node_id:
            lines.extend([
                "Type a new node ID then press Enter to save",
                "",
                f"New Node ID|> {self.new_node_id}|"
            ])
        else:
            self.node_ids: List[str] = self.loader.call_provider(ProviderCall.get_registered_node_ids)
            if len(self.node_ids) > 0:
                lines.append("Registered Node IDs:")
                for i, node_id in enumerate(self.node_ids):
                    l_cursor = "|>" if i == self.select_idx else "  "
                    r_cursor = "<|" if i == self.select_idx else "  "
                    lines.append(f" {l_cursor} {node_id} {r_cursor}")
                lines.append("")
            l_cursor = "|>" if self.select_idx == len(self.node_ids) else "  "
            r_cursor = "<|" if self.select_idx == len(self.node_ids) else "  "
            lines.append(f" {l_cursor} Register new node id {r_cursor}")

        return lines
