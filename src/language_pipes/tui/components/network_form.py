from typing import Callable, Optional

from language_pipes.tui.frame.editor import Editor
from language_pipes.util.config import default_config_dir
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.distributed_state_network.objects.endpoint import Endpoint

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

    def start(self) -> None:
        if not self.loader.provider_available(ProviderCall.get_network_config):
            self.state.set_status("Provider 'get_network_config' unavailable; edit disabled", "info")
            return
        if not self.loader.provider_available(ProviderCall.save_network_config):
            self.state.set_status("Provider 'save_network_config' unavailable; edit disabled", "info")
            return

        try:
            cfg = self.loader.get_network_config()
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
            ]
        )
        
        self.state.set_status("Editing Network -> Configure", "info")

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
            self.loader.save_network_config(payload)
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