from enum import Enum
from typing import Callable, Optional, List, Dict, Any

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.tui.content_provider.network_provider import RouterStatus
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.components.network_form.node_id_editor import NodeIdEditor
from language_pipes.tui.components.network_form.peer_port_editor import PeerPortEditor
from language_pipes.tui.components.network_form.whitelist_editor import WhitelistEditor
from language_pipes.tui.components.network_form.network_ip_editor import NetworkIpEditor
from language_pipes.tui.components.network_form.bootstrap_nodes_editor import (
    BootstrapNodesEditor,
)


class NetworkKeyEditorState(Enum):
    LIST = 0
    INPUT = 1
    SHOW = 2


class NetworkForm:
    confirm: Confirm
    state: FrameState
    loader: ContentLoader

    def __init__(
        self,
        loader: ContentLoader,
        state: FrameState,
        confirm: Confirm,
        change_nav: Callable,
        exit_page: Callable,
        is_focused: Callable,
    ):
        self.state = state
        self.loader = loader
        self.confirm = confirm
        self.change_nav = change_nav
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.edit_fields: List[Dict[str, Optional[Any]]] = []
        self.edit_field_idx = 0
        self.field_editor_visible = False
        self.network_key_select_idx = 0
        self.network_key_max_idx = 1
        self.network_key_state = NetworkKeyEditorState.LIST
        self.network_key = None
        self.network_key_input = ""
        self.network_key_valid = False
        self.node_id_editor = NodeIdEditor(loader, confirm, self.exit_field_editor)
        self.network_ip_editor = NetworkIpEditor(
            loader, confirm, self.exit_field_editor
        )
        self.peer_port_editor = PeerPortEditor(loader, confirm, self.exit_field_editor)
        self.bootstrap_nodes_editor = BootstrapNodesEditor(
            loader, confirm, self.exit_field_editor
        )
        self.whitelist_editor = WhitelistEditor(loader, confirm, self.exit_field_editor)
        self.start()

    def restart_field_editors(self):
        self.node_id_editor.restart()
        self.restart_network_key_editor()
        self.bootstrap_nodes_editor.restart()
        self.network_ip_editor.restart()
        self.peer_port_editor.restart()
        self.whitelist_editor.restart()

    def get_current_field_editor(self):
        res = self.get_current_field()
        if res is None:
            return None
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor
        if current_field == "network_ip":
            return self.network_ip_editor
        if current_field == "peer_port":
            return self.peer_port_editor
        if current_field == "bootstrap_nodes":
            return self.bootstrap_nodes_editor
        if current_field == "whitelist_node_ids":
            return self.whitelist_editor

    def back(self) -> bool:
        res = self.get_current_field_editor()
        if res is None:
            return True
        return res.back()

    def exit_field_editor(self):
        self.field_editor_visible = False
        self.edit_fields = self.get_edit_fields()

    def start(self) -> None:
        if not self.loader.provider_available(ProviderCall.get_network_config):
            self.state.set_status(
                "Provider 'get_network_config' unavailable; edit disabled", "error"
            )
            return
        if not self.loader.provider_available(ProviderCall.save_network_config):
            self.state.set_status(
                "Provider 'save_network_config' unavailable; edit disabled", "error"
            )
            return

        self.edit_fields = self.get_edit_fields()
        self.edit_field_idx = 0
        self.field_editor_visible = False
        self.set_status()

    def get_edit_fields(self) -> List[Dict[str, Optional[Any]]]:
        try:
            cfg: DSNodeConfig = self.loader.call_provider(
                ProviderCall.get_network_config
            )
        except Exception as ex:
            self.state.set_status(f"Failed to load network config: {ex}", "error")
            return []

        key_label = "*" * 10 if cfg.aes_key is not None else ""
        return [
            {
                "name": "node_id",
                "label": "Node ID",
                "value": str(cfg.node_id),
                "error": None,
            },
            {
                "name": "network_key",
                "label": "Netwok Key",
                "value": key_label,
                "error": None,
                "masked": True,
            },
            {
                "name": "network_ip",
                "label": "IP Address",
                "value": cfg.network_ip,
                "error": None,
            },
            {
                "name": "peer_port",
                "label": "Peer Port",
                "value": cfg.port,
                "error": None,
            },
            {
                "name": "bootstrap_nodes",
                "label": "Bootstrap Nodes",
                "value": f"{len(cfg.bootstrap_nodes)} node(s)",
            },
            {
                "name": "whitelist_node_ids",
                "label": "Whitelist",
                "value": f"{len(cfg.whitelist_node_ids)} node(s)",
            },
        ]

    def set_status(self):
        self.state.set_status("Editing Network -> Configure", "info")

    def get_current_field(self) -> Optional[tuple[str, str]]:
        if not self.edit_fields:
            return None

        field = self.edit_fields[self.edit_field_idx]
        field_name = str(field.get("name", ""))
        raw = str(field.get("value", "")).strip()
        return field_name, raw

    def prev_field(self):
        if not self.edit_fields:
            return
        self.edit_field_idx = max(0, self.edit_field_idx - 1)

    def next_field(self):
        if not self.edit_fields:
            return
        self.edit_field_idx = min(len(self.edit_fields) - 1, self.edit_field_idx + 1)

    def enter_field(self):
        if not self.edit_fields:
            return
        self.field_editor_visible = True
        self.restart_field_editors()

    def restart_network_key_editor(self, reset_select: bool = True):
        if reset_select:
            self.network_key_select_idx = 0
        self.network_key_state = NetworkKeyEditorState.LIST
        config: DSNodeConfig = self.loader.call_provider(
            ProviderCall.get_network_config
        )
        self.network_key = config.aes_key
        self.network_key_input = ""
        self.network_key_valid = False
        self.network_key_max_idx = 3 if self.network_key not in (None, "") else 1

    def network_key_back(self) -> bool:
        exit_field_editor = self.network_key_state == NetworkKeyEditorState.LIST
        self.restart_network_key_editor(False)
        return exit_field_editor

    def is_editing_network_key(self) -> bool:
        res = self.get_current_field()
        return res is not None and res[0] == "network_key"

    def get_network_key_footer(self) -> str:
        if self.network_key_state == NetworkKeyEditorState.INPUT:
            return (
                "[A-Z]: Type key   Backspace: delete char   Esc: Back   Enter: Accept"
            )
        if self.network_key_state == NetworkKeyEditorState.SHOW:
            return "Enter/Esc: Back"
        return "Arrows U/D: Change choice   Enter: Confirm   Esc: Back"

    def on_network_key_prev(self):
        self.network_key_select_idx = max(0, self.network_key_select_idx - 1)

    def on_network_key_next(self):
        self.network_key_select_idx = min(
            self.network_key_max_idx, self.network_key_select_idx + 1
        )

    def on_network_key_backspace(self):
        if self.network_key_state != NetworkKeyEditorState.INPUT:
            return
        self.network_key_input = self.network_key_input[:-1]
        self.network_key_valid = self.loader.call_provider(
            ProviderCall.validate_aes_key, self.network_key_input
        )

    def on_network_key_char(self, ch: str):
        if self.network_key_state != NetworkKeyEditorState.INPUT:
            return
        self.network_key_input += ch
        self.network_key_valid = self.loader.call_provider(
            ProviderCall.validate_aes_key, self.network_key_input
        )

    def save_network_key_input(self):
        config: DSNodeConfig = self.loader.call_provider(
            ProviderCall.get_network_config
        )
        config.aes_key = self.network_key_input
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_field_editor()

    def generate_network_key(self):
        config: DSNodeConfig = self.loader.call_provider(
            ProviderCall.get_network_config
        )
        config.aes_key = self.loader.call_provider(ProviderCall.generate_aes_key)
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_field_editor()

    def delete_network_key(self):
        config: DSNodeConfig = self.loader.call_provider(
            ProviderCall.get_network_config
        )
        config.aes_key = None
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_field_editor()

    def on_network_key_enter(self):
        if self.network_key_select_idx == 0:
            if self.network_key_state == NetworkKeyEditorState.LIST:
                self.network_key_state = NetworkKeyEditorState.INPUT
            elif self.network_key_state == NetworkKeyEditorState.INPUT:
                if self.network_key_valid:
                    self.confirm.open(
                        f"Save this network key?\n{self.network_key_input[:32]}\n{self.network_key_input[32:]}",
                        on_apply=self.save_network_key_input,
                        on_discard=lambda: self.restart_network_key_editor(False),
                        confirm_msg=(
                            f"Saved network key!\n{self.network_key_input[:32]}\n"
                            f"{self.network_key_input[32:]}"
                        ),
                    )
        elif self.network_key_select_idx == 1:
            self.confirm.open(
                "Generate a new network key?",
                on_apply=self.generate_network_key,
                on_discard=lambda: self.restart_network_key_editor(False),
                confirm_msg="New network key generated",
            )
        elif self.network_key_select_idx == 2:
            if self.network_key_state == NetworkKeyEditorState.LIST:
                self.network_key_state = NetworkKeyEditorState.SHOW
            elif self.network_key_state == NetworkKeyEditorState.SHOW:
                self.exit_field_editor()
        elif self.network_key_select_idx == 3:
            self.confirm.open(
                "Delete the current network key?\n\nThis will open the network to all\nconnections unless a whitelist is set.",
                on_apply=self.delete_network_key,
                on_discard=lambda: self.restart_network_key_editor(False),
                confirm_msg="Network key deleted",
            )

    def on_network_key(self, key: PressedKey, ch: str = ""):
        if key == PressedKey.ArrowUp:
            self.on_network_key_prev()
        elif key == PressedKey.ArrowDown:
            self.on_network_key_next()
        elif key == PressedKey.Enter:
            self.on_network_key_enter()
        elif key == PressedKey.Alpha:
            self.on_network_key_char(ch)
        elif key == PressedKey.Backspace:
            self.on_network_key_backspace()

    def get_network_key_list_lines(self) -> List[str]:
        lines = ["Editing Network Key"]

        def add_option(option: str, idx: int):
            l_cursor = "|>" if idx == self.network_key_select_idx else "  "
            r_cursor = "<|" if idx == self.network_key_select_idx else "  "
            lines.append(f" {l_cursor} {option} {r_cursor}")

        add_option("Enter Key", 0)
        add_option("Generate New Key", 1)
        if self.network_key_max_idx > 1:
            add_option("Show Existing Key", 2)
            add_option("Delete Existing Key", 3)
        lines.append("")
        return lines

    def get_network_key_input_lines(self) -> List[str]:
        key = (
            self.network_key_input[-40:]
            if len(self.network_key_input) > 40
            else self.network_key_input
        )
        lines = [
            "Enter AES hex key",
            "",
            f"Key: {key}",
            f"Length: {len(self.network_key_input)}",
            "",
        ]
        if self.network_key_valid:
            lines.append("Valid Key!")
        else:
            lines.append("Invalid key: must be a valid aes key hex string")
        return lines

    def get_network_key_show_lines(self) -> List[str]:
        key = self.network_key or ""
        return ["Network Key", "", key[:32], key[32:]]

    def get_network_key_lines(self) -> List[str]:
        if self.network_key_state == NetworkKeyEditorState.LIST:
            return self.get_network_key_list_lines()
        if self.network_key_state == NetworkKeyEditorState.INPUT:
            return self.get_network_key_input_lines()
        if self.network_key_state == NetworkKeyEditorState.SHOW:
            return self.get_network_key_show_lines()
        return []

    def _form_lines(self) -> List[str]:
        lines = ["Edit Network Configuration:"]
        for idx, field in enumerate(self.edit_fields):
            l_cursor = (
                "|>" if idx == self.edit_field_idx and self.is_focused() else "  "
            )
            r_cursor = (
                "<|" if idx == self.edit_field_idx and self.is_focused() else "  "
            )
            name = str(field.get("label", "field"))
            value = str(field.get("value", ""))
            error = field.get("error")
            lines.append(f" {l_cursor} {name}: {value} {r_cursor}")
            if error:
                lines.append(f"    ! {error}")

        tip = ""
        res = self.get_current_field()
        if res is not None and res[0] in TIPS["network"]["configure"]:
            tip = TIPS["network"]["configure"][res[0]]

        lines.extend(["", tip])
        return lines

    def get_view(self) -> List[str]:
        if self.field_editor_visible:
            return self.get_editor_lines()
        if not self.is_focused() and self.loader.provider_available(
            ProviderCall.get_network_config
        ):
            preview = self.show_preview(
                self.loader.call_provider(ProviderCall.get_network_config)
            )
            if preview is not None:
                return preview
        return self._form_lines()

    def get_footer(self) -> str:
        if not self.is_focused():
            return ""
        if not self.field_editor_visible:
            return "Arrows U/D: Change property to edit   Enter: Next   Esc: Back"
        if self.is_editing_network_key():
            return self.get_network_key_footer()
        res = self.get_current_field_editor()
        if res is None:
            return ""
        return res.get_footer()

    def get_editor_lines(self) -> List[str]:
        if self.is_editing_network_key():
            return self.get_network_key_lines()
        res = self.get_current_field_editor()
        if res is None:
            return []
        return res.get_lines()

    def on_key(self, key: PressedKey, ch: str = ""):
        if key == PressedKey.Escape:
            if self.field_editor_visible:
                should_exit = (
                    self.network_key_back()
                    if self.is_editing_network_key()
                    else self.back()
                )
                if should_exit:
                    self.exit_field_editor()
                    self.set_status()
            else:
                self.exit_page()
            return

        if not self.field_editor_visible:
            if key == PressedKey.ArrowUp:
                self.prev_field()
            elif key == PressedKey.ArrowDown:
                self.next_field()
            elif key == PressedKey.Enter:
                self.enter_field()
                field_name = self.edit_fields[self.edit_field_idx]["label"]
                self.state.set_status(f"Editing {field_name}")
            return

        if self.is_editing_network_key():
            return self.on_network_key(key, ch)

        res = self.get_current_field_editor()
        if res is None:
            return
        return res.on_key(key, ch)

    def _open_edit_confirm(
        self,
        message: str,
        *,
        on_apply: Callable[[], None],
        on_discard: Callable[[], None],
    ) -> None:
        self._pending_apply = on_apply
        self._pending_discard = on_discard
        self.confirm.open(message, on_apply, on_discard)

    # Returns string on error
    def validate_current_field(self) -> Optional[str]:
        res = self.get_current_field()
        if res is None:
            return "Not currently editing a form"

        error = None
        field_name, raw = res

        if field_name in ("node_id") and raw == "":
            error = f"{field_name} is required"

        elif field_name == "bootstrap_port":
            try:
                value = int(raw)
                if value < 1 or value > 65535:
                    error = "bootstrap_port must be 1-65535"
            except Exception:
                error = "bootstrap_port must be an integer"

        return error

    def on_exit(self):
        status: RouterStatus = self.loader.call_provider(
            ProviderCall.get_network_status
        )
        if status is not None and status.running:
            return

        def on_apply():
            self.loader.call_provider(ProviderCall.start_network)
            self.change_nav("Network", "Status")

        self.confirm.open(
            "Start connection to network?", on_apply=on_apply, on_discard=lambda: None
        )

    @staticmethod
    def show_preview(payload: DSNodeConfig) -> Optional[List[str]]:
        if not isinstance(payload, DSNodeConfig):
            return None

        key_text = ""
        if payload.aes_key not in (None, ""):
            key_text = "*" * 10

        details = [
            f"- node_id: {payload.node_id}",
            f"- network_key: {key_text}",
            f"- IP Address: {payload.network_ip}",
            f"- Peer Port: {payload.port}",
            f"- Bootstrap Nodes: {len(payload.bootstrap_nodes)} node(s)",
            f"- Whitelist: {len(payload.whitelist_node_ids)} node(s)",
        ]

        return details
