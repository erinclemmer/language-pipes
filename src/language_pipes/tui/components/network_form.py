from typing import Dict, Any

from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader, ProviderCall
from language_pipes.tui.components.exit_confirm import ExitConfirm

class NetworkForm:
    editor: Editor
    state: FrameState
    loader: ContentLoader
    edit_confirm: ExitConfirm

    def __init__(
            self, 
            loader: ContentLoader, 
            state: FrameState, 
            editor: Editor, 
            edit_confirm: ExitConfirm
        ):
        self.state = state
        self.loader = loader
        self.editor = editor
        self.edit_confirm = ExitConfirm

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

    def _build_network_payload(self) -> Dict[str, Any]:
        values = {str(f.get("name")): str(f.get("value", "")).strip() for f in self.editor.edit_fields}
        return {
            "node_id": values.get("node_id", ""),
            "network_key": values.get("network_key", ""),
            "aes_key": values.get("network_key", ""),
            "bootstrap_address": values.get("bootstrap_address", ""),
            "bootstrap_port": int(values.get("bootstrap_port", "0")),
        }

    def submit(self):
        def apply_network() -> None:
            self.loader.save_network_config(payload)
            self._exit_edit_mode()
            self.state.set_status("Saved Network -> Configure", "info")

        self._open_edit_confirm(
            "Apply changes? Network reconnect may take a few seconds.",
            on_apply=apply_network,
            on_discard=self._discard_form,
        )
