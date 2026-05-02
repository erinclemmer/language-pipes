from typing import Callable, Optional, List, Dict, Any

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.network_form.network_key_editor import NetworkKeyEditor
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState

from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.components.network_form.node_id_editor import NodeIdEditor
from language_pipes.tui.components.network_form.peer_port_editor import PeerPortEditor
from language_pipes.tui.components.network_form.whitelist_editor import WhitelistEditor
from language_pipes.tui.components.network_form.network_ip_editor import NetworkIpEditor
from language_pipes.tui.components.network_form.bootstrap_nodes_editor import BootstrapNodesEditor
from language_pipes.tui.util.text import make_footer_text

class NetworkForm:
    confirm: Confirm
    state: FrameState
    provider: ContentProvider

    def __init__(
        self,
        provider: ContentProvider,
        state: FrameState,
        confirm: Confirm,
        change_nav: Callable,
        exit_page: Callable,
        is_focused: Callable,
    ):
        self.state = state
        self.provider = provider
        self.confirm = confirm
        self.change_nav = change_nav
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.edit_field_idx = 0
        self.field_editor_visible = False
        self.network_key_editor = NetworkKeyEditor(provider, confirm, self.exit_field_editor)
        self.node_id_editor = NodeIdEditor(provider, confirm, self.exit_field_editor)
        self.network_ip_editor = NetworkIpEditor(provider, confirm, self.exit_field_editor)
        self.peer_port_editor = PeerPortEditor(provider, confirm, self.exit_field_editor)
        self.bootstrap_nodes_editor = BootstrapNodesEditor(provider, confirm, self.exit_field_editor)
        self.whitelist_editor = WhitelistEditor(provider, confirm, self.exit_field_editor)
        self.start()

    def restart_field_editors(self):
        self.node_id_editor.restart()
        self.bootstrap_nodes_editor.restart()
        self.network_ip_editor.restart()
        self.peer_port_editor.restart()
        self.whitelist_editor.restart()
        self.network_key_editor.restart()

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
        if current_field == "network_key":
            return self.network_key_editor
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
        
    def start(self) -> None:
        self.edit_field_idx = 0
        self.field_editor_visible = False

    def get_edit_fields(self) -> List[Dict[str, Optional[Any]]]:
        cfg = self.provider.network_provider.get_network_config()

        key_label = "*" * 10 if cfg.aes_key is not None else ""
        fields = [
            {
                "name": "node_id",
                "label": "Node ID",
                "value": str(cfg.node_id),
                "error": None if cfg.node_id is not None else "Node ID must be set to start server",
            }
        ]

        if cfg.node_id is not None:
            fields.extend([
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
            ])

        return fields

    def get_current_field(self) -> Optional[tuple[str, str]]:
        field = self.get_edit_fields()[self.edit_field_idx]
        field_name = str(field.get("name", ""))
        raw = str(field.get("value", "")).strip()
        return field_name, raw

    def prev_field(self):
        self.edit_field_idx = max(0, self.edit_field_idx - 1)

    def next_field(self):
        self.edit_field_idx = min(len(self.get_edit_fields()) - 1, self.edit_field_idx + 1)

    def enter_field(self):
        self.field_editor_visible = True
        self.restart_field_editors()

    def _form_lines(self) -> List[str]:
        lines = ["Edit Network Configuration:"]
        for idx, field in enumerate(self.get_edit_fields()):
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
        return self._form_lines()

    def get_footer(self) -> str:
        if not self.is_focused():
            return ""
        if not self.field_editor_visible:
            return make_footer_text(["Arrows U/D: Change property", "Enter: Select", "Esc: Menu"])
        res = self.get_current_field_editor()
        if res is None:
            return ""
        return res.get_footer()

    def get_editor_lines(self) -> List[str]:
        res = self.get_current_field_editor()
        if res is None:
            return []
        return res.get_lines()

    def on_key(self, key: PressedKey, ch: str = ""):
        if key == PressedKey.Escape:
            if self.field_editor_visible:
                should_exit = self.back()
                if should_exit:
                    self.exit_field_editor()
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
            return

        res = self.get_current_field_editor()
        if res is None:
            return
        return res.on_key(key, ch)

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
