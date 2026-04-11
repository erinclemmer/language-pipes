from typing import Any, Callable, List, Optional, Dict, Tuple

from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from language_pipes.tui.content_provider.job_provider import MetaJob
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.content_provider.network_provider import RouterStatus
from language_pipes.tui.components.hosted_models_view import format_pipe_strings
from language_pipes.tui.content_provider.model_provider import ModelStatus, ModelToLoad, ModelStatusInfo

class Dashboard:
    def _get_options(self) -> List[str]:
        opts = []

        if self.router_status is None or self.router_status.state == "stopped":
            if self.network_config.node_id != "" and self.network_port_available():
                opts.append("Start Network Server")
            opts.append("Configure Network Server")

        if self.router_status is not None and self.router_status.state == "running":
            opts.append("Stop Network Server")
            opts.append("Configure Network Server")
            if self.job_serv_running:
                opts.append("Stop Job Server")
            elif self.job_port_available():
                opts.append("Start Job Server")
            opts.append("Configure Job Server")
            if len(self.models_to_load) > 0:
                if self._has_active_model():
                    opts.append("Unload Models")
                else:
                    opts.append("Load Models")

                opts.append("Configure Models")

            opts.append("Show Logs")

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
    oai_port: int
    job_serv_running: bool
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
        self.oai_port = self.loader.call_provider(ProviderCall.get_oai_port)
        self.job_serv_running = False
        self.models_to_load = []
        self.models_status = { }

    def network_port_available(self) -> bool:
        if self.network_config is None:
            return True
        return self.loader.call_provider(ProviderCall.is_port_available, self.network_config.port)

    def job_port_available(self) -> bool:
        return self.loader.call_provider(ProviderCall.is_port_available, self.oai_port)

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
        elif selected_option == "Start Job Server":
            api_keys = self.loader.call_provider(ProviderCall.get_api_keys)
            self.loader.call_provider(ProviderCall.start_oai_server, (self.oai_port, api_keys))
        elif selected_option == "Stop Job Server":
            self.loader.call_provider(ProviderCall.stop_oai_server)
        elif selected_option == "Configure Job Server":
            self.change_nav("Jobs", "Server")
        elif selected_option == "Show Logs":
            self.change_nav("Home", "Activity")

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
            peer_text = f"\n{getattr(self.router_status, 'num_peers', 0)} peer(s) connected"
        
        state_label = "Off"
        if self.router_status is not None:
            if  self.router_status.state == "running":
                state_label = f"Running\n{self.network_config.node_id}@{self.network_config.network_ip}:{self.router_status.port}"
            else:
                state_label = self.router_status.state
        
        return f"{state_label}{peer_text}"

    def _format_model_line(self, model: ModelToLoad, running: List[ModelStatusInfo], jobs: List[MetaJob]) -> List[str]:
        ends_string = "+ ends" if model.load_ends else ""
        lines = [
            f"{model.model_id} ({model.max_memory}GB) {ends_string}"
        ]
        
        lines.extend(format_pipe_strings(running))
        for j in jobs:
            lines.extend([
                f"Job {j.job_id[:4]} from {j.origin_node_id}, Token: {j.current_token}"
            ])

        return lines

    def get_view(self) -> Tuple[List[str], List[str]]:
        self.models_to_load = self.loader.call_provider(ProviderCall.get_models_to_load)
        self.network_config = self.loader.call_provider(ProviderCall.get_network_config)
        self.router_status = self.loader.call_provider(ProviderCall.get_network_status)
        self.job_serv_running = self.loader.call_provider(ProviderCall.oai_server_running)
        self.oai_port = self.loader.call_provider(ProviderCall.get_oai_port)

        lines = ["   Options:", ""]
        right_panel = [self._get_ram_usage(), ""]

        job_str = f"running on port {self.oai_port}" if self.job_serv_running else "stopped"
        if self.router_status is not None or self.network_port_available():
            if self.network_config.node_id != "":
                right_panel.extend([f"Network: {self._get_network_label()}", ""])
            else:
                right_panel.extend(["Warning:\nNode ID not set, cannot start network", ""])
        else:
            right_panel.extend([f"Warning:\nNetwork port {self.network_config.port} not available", ""])

        if self.router_status is not None and self.router_status.state == "running":
            if self.job_serv_running or self.job_port_available():
                right_panel.extend([f"Job Server: {job_str}", ""])
            else:
                right_panel.extend([f"Warning:\nJob port {self.oai_port} is not available", ""])

        selected_option = self._get_selected_option()
        for label in self._get_options():
            selected = self.is_focused() and label == selected_option
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.extend([f"{l_cursor} {label} {r_cursor}", ""])
        lines.extend(["", ""])

        self.models_status = self.loader.call_provider(ProviderCall.get_models_status)
        jobs: List[MetaJob] = self.loader.call_provider(ProviderCall.get_active_jobs)
        for model in self.models_to_load:
            model_statuses = self.models_status.get(model.model_id, [])
            pipe_ids = [p.pipe_id for p in model_statuses]
            model_jobs = [j for j in jobs if j.pipe_id in pipe_ids]
            right_panel.extend(self._format_model_line(model, model_statuses, model_jobs))
        
        return lines, right_panel

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
