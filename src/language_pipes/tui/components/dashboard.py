from typing import Any, Callable, List, Optional, Dict, Tuple

from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.content_provider.network_provider import RouterStatus
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.tui.content_provider.model_provider import ModelStatus, ModelToLoad, ModelStatusInfo

class Dashboard:
    def _get_options(self) -> List[str]:
        opts = []
        if self.router_status is None or self.router_status.state == "stopped":
            opts.append("Start Network Server")

        if self.router_status is not None and self.router_status.state == "running":
            opts.append("Stop Network Server")

        opts.append("Configure Network Server")

        if len(self.models_to_load) > 0:
            if self._has_active_model():
                opts.append("Unload Models")
            else:
                opts.append("Load Models")

        opts.append("Configure Models")

        return opts

    def _get_selected_option(self) -> str:
        return self._get_options()[self.selected_idx]

    def _get_num_options(self) -> int:
        return len(self._get_options())

    def _has_active_model(self) -> bool:
        for key in self.models_status.keys():
            for status in self.models_status[key]:
                if status.status == ModelStatus.Running:
                    return True
        return False

    network_config: DSNodeConfig
    router_status: Optional[RouterStatus]

    selected_idx: int
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
        self.router_status = None
        self.network_config = self.loader.call_provider(ProviderCall.get_network_config)
        self.selected_idx = 0
        self.models_to_load = []
        self.models_status = { }

    def on_prev(self):
        self.selected_idx -= 1
        if self.selected_idx < 0:
            self.selected_idx = self._get_num_options() - 1
    
    def on_next(self):
        self.selected_idx += 1
        if self.selected_idx >= self._get_num_options():
            self.selected_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
                self.on_enter()
        elif key == PressedKey.Escape:
            self.exit_page()

    def on_enter(self):
        selected_option = self._get_selected_option()
        if selected_option == "Start Network Server":
            self.loader.call_provider(ProviderCall.start_network)
        elif selected_option == "Stop Network Server":
            self.loader.call_provider(ProviderCall.stop_network)
        elif selected_option == "Configure Network Server":
            self.change_nav("Network", "Configure")
        elif selected_option == "Load Models":
            for model in self.models_to_load:
                self.loader.call_provider(ProviderCall.host_model, model)
        elif selected_option == "Unload Models":
            for model in self.models_to_load:
                self.loader.call_provider(ProviderCall.shutdown_models, model.model_id)
        elif selected_option == "Configure Models":
            self.change_nav("Models", "Hosted")

    def _get_ram_usage(self) -> str:
        used_ram = self.loader.call_provider(ProviderCall.get_used_system_ram)
        total_ram = self.loader.call_provider(ProviderCall.get_total_system_ram)
        
        return f"System RAM: {used_ram:.1f}/{total_ram:.1f}GB"

    @staticmethod
    def _get_state(status: Optional[RouterStatus]) -> str:
        if status is None:
            return "stopped"
        return status.state
        
    def _get_network_label(self):
        peer_text = ""
        if self.router_status is not None and self.router_status.state == "running":
            peer_text = f" ({getattr(self.router_status, 'num_peers', 0)} peer(s) connected)"
        
        state_label = "Off"
        if self.router_status is not None:
            if  self.router_status.state == "running":
                state_label = f"{self.config.node_id}@{self.config.network_ip}:{self.router_status.port}"
            else:
                state_label = self.router_status.state
        
        return f"{state_label}{peer_text}"

    def get_view(self) -> Tuple[List[str], List[str]]:
        self.models_to_load = self.loader.call_provider(ProviderCall.get_models_to_load)
        self.config = self.loader.call_provider(ProviderCall.get_network_config)
        self.router_status = self.loader.call_provider(ProviderCall.get_network_status)

        lines = [self._get_ram_usage(), ""]
        if self.config.node_id != "":
            lines.extend([f"Network: {self._get_network_label()}", ""])
        else:
            lines.extend(["Warning: Node ID not set, cannot start network server", ""])

        selected_option = self._get_selected_option()
        for label in self._get_options():
            selected = self.is_focused() and label == selected_option
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        lines.extend(["", ""])

        self.models_status = self.loader.call_provider(ProviderCall.get_models_status)
        for model in self.models_to_load:
            model_statuses = self.models_status.get(model.model_id, [])
            lines.append(format_model_line(model, selected=False, running=model_statuses))
        
        return lines, ["test"]

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
