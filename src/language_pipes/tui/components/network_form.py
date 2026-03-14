from typing import Callable, Optional, List

from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.util.config import default_config_dir
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
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
        self.registering_node_id = False
        self.select_idx = 0
        self.node_ids = []

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
            def save_node_id():
                config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
                config.node_id = self.new_node_id
                self.loader.call_provider(ProviderCall.save_new_node_id, self.new_node_id)
                self.loader.call_provider(ProviderCall.save_network_config, config)
                self.confirm.close()
                self.exit_editor()
            self.confirm.open(f"Register \"{self.new_node_id}\"?", save_node_id, discard_choice)
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
            self.confirm.open(f"Use \"{selected_node_id}\"?", use_node_id, discard_choice)

    def on_char(self, ch: str):
        self.new_node_id += ch

    def on_backspace(self):
        self.new_node_id = self.new_node_id[:-1]

    def get_lines(self):
        lines = []

        if self.registering_node_id:
            lines.append(f"New Node ID: {self.new_node_id}")
        else:
            self.node_ids: List[str] = self.loader.call_provider(ProviderCall.get_registered_node_ids)
            if len(self.node_ids) > 0:
                lines.append("Registered Node IDs:")
                for i, node_id in enumerate(self.node_ids):
                    cursor_txt = " >" if i == self.select_idx else "  "
                    lines.append(f"{cursor_txt}{node_id}")
                lines.append("")
            cursor_txt = " >" if self.select_idx == len(self.node_ids) else "  "
            lines.append(f"{cursor_txt}Register new node id")

        return lines

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
            confirm: Confirm
        ):
        self.state = state
        self.loader = loader
        self.editor = editor
        self.confirm = confirm
        self.node_id_editor = NodeIdEditor(loader, confirm, self.exit_field_editor)

    def exit_field_editor(self):
        self.editor.field_editor_visible = False
        self.node_id_editor.restart()
        self.start()

    def start(self) -> None:
        if not self.loader.provider_available(ProviderCall.get_network_config):
            self.state.set_status("Provider 'get_network_config' unavailable; edit disabled", "error")
            return
        if not self.loader.provider_available(ProviderCall.save_network_config):
            self.state.set_status("Provider 'save_network_config' unavailable; edit disabled", "error")
            return

        try:
            cfg = self.loader.call_provider(ProviderCall.get_network_config)
        except Exception as ex:
            self.state.set_status(f"Failed to load network config: {ex}", "error")
            return

        bootstrap_address = cfg.bootstrap_nodes[0].address if len(cfg.bootstrap_nodes) > 0 else ""
        bootstrap_port = str(cfg.bootstrap_nodes[0].port) if len(cfg.bootstrap_nodes) > 0 else ""

        self.editor.start_edit_mode(
            form_name="network_config",
            edit_fields=[
                {"name": "node_id", "value": str(cfg.node_id), "error": None},
                {"name": "network_key", "value": str(cfg.aes_key), "error": None, "masked": True},
                {
                    "name": "bootstrap_address",
                    "value": bootstrap_address,
                    "error": None,
                },
                {"name": "bootstrap_port", "value": str(bootstrap_port), "error": None},
            ],
            form=self
        )
        
        self.state.set_status("Editing Network -> Configure", "info")

    def get_editor_lines(self) -> List[str]:
        res = self.editor.get_current_field()
        if res is None: 
            return []
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor.get_lines()
        return []

    def on_key(self, key: PressedKey, ch: str = ""):
        res = self.editor.get_current_field()
        if res is None:
            return
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor.on_key(key, ch)

    def _build_payload(self) -> DSNodeConfig:
        values = {str(f.get("name")): str(f.get("value", "")).strip() for f in self.editor.edit_fields}
        
        data = {
            "node_id": values.get("node_id", ""),
            "aes_key": values.get("network_key", ""),
            "bootstrap_address": values.get("bootstrap_address", ""),
            "bootstrap_port": int(values.get("bootstrap_port", "0")),
        }

        return DSNodeConfig(
            node_id=data["node_id"],
            aes_key=data["aes_key"],
            credential_dir=default_config_dir() + "/credentials",
            port=5000,
            network_ip="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[
                Endpoint(data["bootstrap_address"], data["bootstrap_port"])
            ] if data["bootstrap_address"] != "" else []
        )

    def submit(self):
        payload = self._build_payload()
        def apply_network() -> None:
            self.loader.call_provider(ProviderCall.save_network_config, payload)
            self.editor.exit_edit_mode()
            self.state.set_status("Saved Network -> Configure", "info")

        def discard_network():
            self.state.set_status("Discarded edits", "info")
            self.editor.discard_form()

        self._open_edit_confirm(
            "Apply changes? Network reconnect may take a few seconds.",
            on_apply=apply_network,
            on_discard=discard_network,
        )

    def _open_edit_confirm(
        self,
        message: str,
        *,
        on_apply: Callable[[], None],
        on_discard: Optional[Callable[[], None]],
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
    
    def on_press_enter(self):
        pass

    @staticmethod
    def show_preview(payload: DSNodeConfig) -> Optional[List[str]]:
        if not isinstance(payload, DSNodeConfig):
            return None

        key_text = ""
        if payload.aes_key not in (None, ""):
            key_text = "*" * len(str(payload.aes_key))

        details = [
            f"- node_id: {payload.node_id}",
            f"- network_key: {key_text}"
        ]

        if len(payload.bootstrap_nodes) > 0:
            details.extend([
                f"- bootstrap_address: {payload.bootstrap_nodes[0].address}",
                f"- bootstrap_port: {payload.bootstrap_nodes[0].port}",
            ])

        return details