from typing import Any, Callable, List, Optional, Dict

from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.content_provider.model_provider import ModelToLoad, ModelStatusInfo


class Dashboard:
    def _has_node_id(self) -> bool:
        config = self._get_config()
        return config is not None and config.node_id != ""

    def _get_options(self, state: str) -> List[tuple]:
        if state == "running":
            options = [("Stop Network Server", "Network", "Status")]
        elif state == "starting":
            options = [("Starting Network Server", "Network", "Status")]
        elif state == "stopping":
            options = [("Stopping Network Server", "Network", "Status")]
        elif self._has_node_id():
            options = [("Start Network Server", "Network", "Status")]
        else:
            options = [("Configure Network", "Network", "Configure")]

        if state == "running":
            options.append(("Host Models", "Models", "Hosted"))
        return options

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

    def on_key(self, key: PressedKey, ch: str):
        status = self._get_status()
        state = self._get_state(status)
        options = self._get_options(state)
        if key == PressedKey.ArrowUp:
            self.selected_idx = (self.selected_idx - 1) % len(options)
        elif key == PressedKey.ArrowDown:
            self.selected_idx = (self.selected_idx + 1) % len(options)
        elif key == PressedKey.Enter:
            if self.selected_idx == 0:
                config = self._get_config()
                node_id = config.node_id if config else ""

                if node_id == "":
                    # Navigate to Network -> Configure page
                    self.change_nav("Network", "Configure")
                elif state == "stopped":
                    self.loader.call_provider(ProviderCall.start_network)
                elif state == "running":
                    self.loader.call_provider(ProviderCall.stop_network)
            else:
                # Host all models in models_to_load list directly
                models_to_load = self._get_models_to_load()
                for model in models_to_load:
                    self.loader.call_provider(ProviderCall.host_model, model)
        elif key == PressedKey.Escape:
            self.exit_page()

    def _get_status(self) -> Optional[Any]:
        try:
            return self.loader.call_provider(ProviderCall.get_network_status)
        except LookupError:
            return None

    def _get_config(self) -> Optional[DSNodeConfig]:
        try:
            return self.loader.call_provider(ProviderCall.get_network_config)
        except LookupError:
            return None

    def _get_models_to_load(self) -> List[Any]:
        try:
            return self.loader.call_provider(ProviderCall.get_models_to_load)
        except LookupError:
            return []

    def _get_ram_usage(self) -> Optional[str]:
        try:
            used_ram = self.loader.call_provider(ProviderCall.get_used_system_ram)
            total_ram = self.loader.call_provider(ProviderCall.get_total_system_ram)
        except LookupError:
            return None
        return f"System RAM: {used_ram:.1f} / {total_ram:.1f} GB"

    @staticmethod
    def _get_state(status: Optional[Any]) -> str:
        if status is None:
            return "stopped"
        if hasattr(status, "state"):
            return getattr(status, "state")
        return "running" if getattr(status, "running", False) else "stopped"

    @staticmethod
    def _get_state_label(state: str) -> str:
        if state == "running":
            return "On"
        if state == "stopped":
            return "Off"
        return state.title()

    def get_view(self) -> List[str]:
        status = self._get_status()
        models_to_load = self._get_models_to_load()
        ram_usage = self._get_ram_usage()
        state = self._get_state(status)
        is_running = state == "running"
        focused = self.is_focused()
        peer_text = (
            f" ({getattr(status, 'num_peers', 0)} peer(s) connected)"
            if is_running
            else ""
        )
        lines = [f"Network Server: {self._get_state_label(state)}{peer_text}", ""]
        if not self._has_node_id():
            lines.extend(["Warning: Node ID not set, cannot start server", ""])
        if ram_usage is not None:
            lines.extend([ram_usage, ""])
        options = self._get_options(state)
        for idx, (label, _, _) in enumerate(options):
            selected = focused and idx == self.selected_idx
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        lines.extend(["", "Hosted Models", ""])
        # Get model status like in the Models -> Hosted page
        try:
            models_status: Dict[str, List[ModelStatusInfo]] = self.loader.call_provider(ProviderCall.get_models_status)
        except LookupError:
            models_status = {}
        for model in models_to_load:
            model_statuses = models_status.get(model.model_id, [])
            lines.append(format_model_line(model, running=model_statuses))
        
        return lines

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
