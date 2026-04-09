from typing import Any, Callable, List, Optional, Dict

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.content_provider.network_provider import RouterStatus
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.tui.content_provider.model_provider import ModelToLoad, ModelStatusInfo

class Dashboard:
    def _has_node_id(self) -> bool:
        config = self.loader.call_provider(ProviderCall.get_network_config)
        return config is not None and config.node_id != ""

    def _get_options(self, status: Optional[RouterStatus]) -> List[str]:
        if status is None:
            return ["Start Network Server"]
        
        state = status.state
        if state == "running":
            options = ["Stop Network Server"]
        elif state == "starting":
            options = ["Starting Network Server"]
        elif state == "stopping":
            options = ["Stopping Network Server"]
        elif self._has_node_id():
            options = ["Start Network Server", "Network", "Status"]
        else:
            options = ["Configure Network", "Network", "Configure"]

        return options

    models_to_load: List[ModelToLoad]
    models_status: Dict[str, List[ModelStatusInfo]]

    def __init__(
        self,
        loader: Any,
        exit_page: Callable,
        is_focused: Callable,
        change_nav: Callable,
    ):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.change_nav = change_nav
        self.selected_idx = 0
        self.models_to_load = []
        self.models_status = { }

    def on_prev(self):
        self.selected_idx -= 1
        if self.selected_idx < 0:
            self.selected_idx = len(self.models_to_load)
    
    def on_next(self):
        self.selected_idx += 1
        if self.selected_idx > len(self.models_to_load):
            self.selected_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        status = self.loader.call_provider(ProviderCall.get_network_status)
        state = self._get_state(status)
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            if self.selected_idx == 0:
                config = self.loader.call_provider(ProviderCall.get_network_config)
                node_id = config.node_id if config else ""

                if node_id == "":
                    # Navigate to Network -> Configure page
                    self.change_nav("Network", "Configure")
                elif state == "stopped":
                    self.loader.call_provider(ProviderCall.start_network)
                elif state == "running":
                    self.loader.call_provider(ProviderCall.stop_network)
            else:
                model = self.models_to_load[self.selected_idx - 1]
                status = self.models_status.get(model.model_id)
                if status:
                    self.loader.call_provider(ProviderCall.shutdown_models, model.model_id)
                else:
                    self.loader.call_provider(ProviderCall.host_model, model)
        elif key == PressedKey.Escape:
            self.exit_page()

    def _get_ram_usage(self) -> Optional[str]:
        try:
            used_ram = self.loader.call_provider(ProviderCall.get_used_system_ram)
            total_ram = self.loader.call_provider(ProviderCall.get_total_system_ram)
        except LookupError:
            return None
        
        return f"System RAM: {used_ram:.1f}/{total_ram:.1f}GB"

    @staticmethod
    def _get_state(status: Optional[RouterStatus]) -> str:
        if status is None:
            return "stopped"
        return status.state
        
    def _get_network_label(self, status: Optional[RouterStatus]):
        peer_text = ""
        if status is not None and status.state == "running":
            peer_text = f" ({getattr(status, 'num_peers', 0)} peer(s) connected)"
        
        state_label = "Off"
        if status is not None:
            if  status.state == "running":
                state_label = f"Running on port {status.port}"
            else:
                state_label = status.state
        
        return f"{state_label}{peer_text}"

    def get_view(self) -> List[str]:
        self.models_to_load = self.loader.call_provider(ProviderCall.get_models_to_load)
        config = self.loader.call_provider(ProviderCall.get_network_config)
        status = self.loader.call_provider(ProviderCall.get_network_status)

        lines = ["Network Server:", f"Status: {self._get_network_label(status)}"]
        if config is not None and self._has_node_id():
            lines.extend([f"Node ID: {config.node_id}"])
        else:
            lines.extend(["Warning: Node ID not set, cannot start server", ""])
        
        ram_usage = self._get_ram_usage()
        if ram_usage is not None:
            lines.extend([ram_usage, ""])

        for idx, label in enumerate(self._get_options(status)):
            selected = self.is_focused() and idx == self.selected_idx
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        lines.extend(["", ""])

        self.models_status = self.loader.call_provider(ProviderCall.get_models_status)
        for i, model in enumerate(self.models_to_load):
            model_statuses = self.models_status.get(model.model_id, [])
            lines.append(format_model_line(model, selected=self.is_focused() and i + 1 == self.selected_idx, running=model_statuses))
        
        return lines

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
