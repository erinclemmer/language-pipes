from typing import Callable, Optional, List, Dict, Any

from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.components.network_form.node_id_editor import NodeIdEditor
from language_pipes.tui.components.network_form.peer_port_editor import PeerPortEditor
from language_pipes.tui.components.network_form.whitelist_editor import WhitelistEditor
from language_pipes.tui.components.network_form.network_ip_editor import NetworkIpEditor
from language_pipes.tui.components.network_form.network_key_editor import NetworkKeyEditor
from language_pipes.tui.components.network_form.bootstrap_nodes_editor import BootstrapNodesEditor

class NetworkForm:
    editor: Editor
    confirm: Confirm
    state: FrameState
    loader: ContentLoader

    def __init__(
            self, 
            loader: ContentLoader, 
            state: FrameState, 
            editor: Editor, 
            confirm: Confirm,
            change_nav: Callable
        ):
        self.state = state
        self.loader = loader
        self.editor = editor
        self.confirm = confirm
        self.change_nav = change_nav
        self.node_id_editor = NodeIdEditor(loader, confirm, self.exit_field_editor)
        self.network_key_editor = NetworkKeyEditor(loader, confirm, self.exit_field_editor)
        self.network_ip_editor = NetworkIpEditor(loader, confirm, self.exit_field_editor)
        self.peer_port_editor = PeerPortEditor(loader, confirm, self.exit_field_editor)
        self.bootstrap_nodes_editor = BootstrapNodesEditor(loader, confirm, self.exit_field_editor)
        self.whitelist_editor = WhitelistEditor(loader, confirm, self.exit_field_editor)

    def restart_field_editors(self):
        self.node_id_editor.restart()
        self.network_key_editor.restart()
        self.bootstrap_nodes_editor.restart()
        self.network_ip_editor.restart()
        self.peer_port_editor.restart()
        self.whitelist_editor.restart()

    def get_current_field_editor(self):
        res = self.editor.get_current_field()
        if res is None: 
            return None
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor
        if current_field == "network_key":
            return self.network_key_editor
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
        self.editor.field_editor_visible = False
        self.editor.edit_fields = self.get_edit_fields()

    def start(self) -> None:
        if not self.loader.provider_available(ProviderCall.get_network_config):
            self.state.set_status("Provider 'get_network_config' unavailable; edit disabled", "error")
            return
        if not self.loader.provider_available(ProviderCall.save_network_config):
            self.state.set_status("Provider 'save_network_config' unavailable; edit disabled", "error")
            return

        
        self.editor.start_edit_mode(
            form_name="network_config",
            edit_fields=self.get_edit_fields(),
            form=self
        )
        res = self.get_current_field_editor()
        if res is not None:
            res.restart()
        self.set_status()

    def get_edit_fields(self) -> List[Dict[str, Optional[Any]]]:
        try:
            cfg: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        except Exception as ex:
            self.state.set_status(f"Failed to load network config: {ex}", "error")
            return []

        key_label = "*" * 10 if cfg.aes_key is not None else ""
        return [
            {"name": "node_id", "label": "Node ID", "value": str(cfg.node_id), "error": None},
            {"name": "network_key", "label": "Netwok Key", "value": key_label, "error": None, "masked": True},
            {"name": "network_ip", "label": "IP Address", "value": cfg.network_ip, "error": None},
            {"name": "peer_port", "label": "Peer Port", "value": cfg.port, "error": None},
            {"name": "bootstrap_nodes", "label": "Bootstrap Nodes", "value": f"{len(cfg.bootstrap_nodes)} node(s)"},
            {"name": "whitelist_node_ids", "label": "Whitelist", "value": f"{len(cfg.whitelist_node_ids)} node(s)"}
        ]
    
    def set_status(self):
        self.state.set_status("Editing Network -> Configure", "info")

    def get_footer(self) -> str:
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
        res = self.editor.get_current_field()
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
        def on_apply():
            self.loader.call_provider(ProviderCall.start_network)
            self.change_nav("Network", "Status")

        self.confirm.open(
            "Start connection to network?",
            on_apply=on_apply,
            on_discard=lambda:None
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
            f"- Whitelist: {len(payload.whitelist_node_ids)} node(s)"
        ]

        return details