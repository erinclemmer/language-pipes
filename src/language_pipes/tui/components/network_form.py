from enum import Enum
from typing import Callable, Optional, List

from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.util.config import default_config_dir
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.components.node_id_editor import NodeIdEditor
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint

class NetworkKeyEditorState(Enum):
    LIST = 0
    INPUT = 1
    GENERATE = 2
    SHOW = 3
    DELETE = 4
    
class NetworkKeyEditor:
    confirm: Confirm
    loader: ContentLoader
    exit_editor: Callable

    key_input: str
    max_idx: int
    select_idx: int
    state: NetworkKeyEditorState

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.loader = loader
        self.confirm = confirm
        self.exit_editor = exit_editor
        self.restart()

    def restart(self):
        self.select_idx = 0
        self.state = NetworkKeyEditorState.LIST
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        self.network_key = config.aes_key
        self.key_input = ""
        self.key_valid = False
        self.max_idx = 3 if self.network_key is not None and self.network_key != '' else 1

    def get_footer(self):
        return "Arrows: Navigate   Enter: Confirm   Esc: Discard"

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Alpha:
            self.on_char(ch)
        elif key == PressedKey.Backspace:
            self.on_backspace()

    def on_backspace(self):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input = self.key_input[:-1]

    def on_char(self, ch: str):
        if self.state != NetworkKeyEditorState.INPUT:
            return
        self.key_input += ch
        self.key_valid = self.loader.call_provider(ProviderCall.validate_aes_key, self.key_input)

    def on_prev(self):
        self.select_idx -= 1
        if self.select_idx < 0:
            self.select_idx = 0
    
    def on_next(self):
        self.select_idx += 1
        if self.select_idx > self.max_idx:
            self.select_idx = self.max_idx

    def on_enter(self):
        if self.select_idx == 0:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.INPUT
            elif self.state == NetworkKeyEditorState.INPUT:
                if self.key_valid:
                    self.confirm.open(
                        f"Save this network key?\n{self.key_input[:32]}\n{self.key_input[32:]}",
                        on_apply=self.save_key_input,
                        on_discard=self.exit_editor,
                        confirm_msg=f"Saved network key!\n{self.key_input[:32]}\n{self.key_input[32:]}"
                    )
        elif self.select_idx == 1:
            self.confirm.open(
                "Generate a new network key?",
                on_apply=self.generate_key,
                on_discard=self.exit_editor,
                confirm_msg="New network key generated"
            )
        elif self.select_idx == 2:
            if self.state == NetworkKeyEditorState.LIST:
                self.state = NetworkKeyEditorState.SHOW
            elif self.state == NetworkKeyEditorState.SHOW:
                self.exit_editor()

    def generate_key(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.aes_key = self.loader.call_provider(ProviderCall.generate_aes_key)
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

    def save_key_input(self):
        config: DSNodeConfig = self.loader.call_provider(ProviderCall.get_network_config)
        config.aes_key = self.key_input
        self.loader.call_provider(ProviderCall.save_network_config, config)
        self.exit_editor()

    def get_list_lines(self) -> List[str]:
        lines = ["Editing Network Key"]
        def add_option(option: str, idx: int):
            l_cursor = "|>" if idx == self.select_idx else "  "
            r_cursor = "<|" if idx == self.select_idx else "  "
            lines.append(f" {l_cursor} {option} {r_cursor}")
        add_option("Enter Key", 0)
        add_option("Generate New Key", 1)
        if self.max_idx > 1:
            add_option("Show Existing Key", 2)
            add_option("Delete Existing Key", 3)
        lines.append("")
        return lines
    
    def get_input_lines(self) -> List[str]:
        key = self.key_input[-40:] if len(self.key_input) > 40 else self.key_input
        lines = [
            "Enter AES hex key",
            "",
            f"Key: {key}",
            f"Length: {len(self.key_input)}",
            ""
        ]

        if self.key_valid:
            lines.append("Valid Key!")
        else:
            lines.append("Invalid key: must be a valid aes key hex string")

        return lines

    def get_show_lines(self) -> List[str]:
        key = ""
        if self.network_key is not None:
            key = self.network_key
        lines = [
            "Network Key",
            "",
            key[:32],
            key[32:]
        ]

        return lines

    def get_lines(self) -> List[str]:
        if self.state == NetworkKeyEditorState.LIST:
            return self.get_list_lines()
        if self.state == NetworkKeyEditorState.INPUT:
            return self.get_input_lines()
        if self.state == NetworkKeyEditorState.SHOW:
            return self.get_show_lines()
        return []

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
        self.network_key_editor = NetworkKeyEditor(loader, confirm, self.exit_field_editor)

    def enter_field_editor(self):
        self.node_id_editor.restart()
        self.network_key_editor.restart()

    def exit_field_editor(self):
        self.editor.field_editor_visible = False
        self.node_id_editor.restart()
        self.network_key_editor.restart()
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
                {"name": "node_id", "label": "Node ID", "value": str(cfg.node_id), "error": None},
                {"name": "network_key", "label": "Netwok Key", "value": str(cfg.aes_key)[:10], "error": None, "masked": True},
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

    def get_footer(self) -> str:
        res = self.editor.get_current_field()
        if res is None: 
            return ""
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor.get_footer()
        if current_field == "network_key":
            return self.network_key_editor.get_footer()
        return ""

    def get_editor_lines(self) -> List[str]:
        res = self.editor.get_current_field()
        if res is None: 
            return []
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor.get_lines()
        if current_field == "network_key":
            return self.network_key_editor.get_lines()
        return []

    def on_key(self, key: PressedKey, ch: str = ""):
        res = self.editor.get_current_field()
        if res is None:
            return
        current_field, _ = res
        if current_field == "node_id":
            return self.node_id_editor.on_key(key, ch)
        if current_field == "network_key":
            return self.network_key_editor.on_key(key, ch)

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