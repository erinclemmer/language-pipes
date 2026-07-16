from typing import Callable, List, Optional, Dict, Set, Tuple

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.job_provider import DEFAULT_JOB_PORT
from language_pipes.distributed_state_network.objects.config import DSNodeConfig
from ansinout import PressedKey
from language_pipes.content_provider.network_provider import RouterStatus
from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.tui.components.view_pipe import format_pipe_view
from language_pipes.content_provider.model_provider import ModelProvider, ModelStatus, ModelToLoad, ModelStatusInfo
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text

class Dashboard:
    def _get_options(self) -> List[str]:
        run_opts = []
        config_opts = []

        if self.router_status is None or self.router_status.state == "stopped":
            if self.network_config.node_id is not None and self.network_config.node_id != "" and self.network_port_available():
                run_opts.append("Start Network Server")
            config_opts.append("Configure Network Server")

        if self.router_status is not None and self.router_status.state == "running":
            run_opts.append("Stop Network Server")
            config_opts.append("Configure Network Server")
            if self.job_serv_running:
                run_opts.append("Stop Job Server")
            elif self.job_port_available():
                run_opts.append("Start Job Server")
            config_opts.append("Configure Job Server")

            config_opts.append("Configure Layer Models")
            config_opts.append("Configure End Models")

            config_opts.append("Show Logs")

        run_opts.extend(config_opts)

        return run_opts

    def _get_selected_option(self) -> Optional[str]:
        opts = self._get_options()
        if len(opts) - 1 < self.selected_idx:
            return None
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
    job_port: Optional[int]
    job_serv_running: bool
    connected_pipes: List[MetaPipe]
    layer_models: List[ModelToLoad]
    models_status: Dict[str, List[ModelStatusInfo]]

    def __init__(
        self,
        provider: ContentProvider,
        exit_page: Callable,
        change_nav: Callable,
    ):
        self.provider = provider
        self.exit_page = exit_page
        self.change_nav = change_nav
        self.router_status = None
        self.network_config = self.provider.network_provider.get_network_config()
        self.selected_idx = 0
        self.pipe_idx = 0
        self.job_port = self.provider.job_provider.get_job_port()
        self.job_serv_running = False
        self.layer_models = []
        self.connected_pipes = []
        self.models_status = { }

    def network_port_available(self) -> bool:
        if self.network_config is None:
            return True
        return ContentProvider.is_port_available(self.network_config.port)

    def job_port_available(self) -> bool:
        port = self.job_port if self.job_port is not None else DEFAULT_JOB_PORT
        return ContentProvider.is_port_available(port)

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
        elif key == PressedKey.PageUp:
            self.on_pg_up()
        elif key == PressedKey.PageDown:
            self.on_pg_down()

    def on_pg_up(self):
        self.pipe_idx = (self.pipe_idx - 1) % len(self.connected_pipes)

    def on_pg_down(self):
        self.pipe_idx = (self.pipe_idx + 1) % len(self.connected_pipes)

    def on_enter(self):
        selected_option = self._get_selected_option()
        if selected_option is None:
            return
        
        if selected_option == "Start Network Server":
            self.provider.network_provider.start_network()
        elif selected_option == "Stop Network Server":
            self.provider.stop_network()
        elif selected_option == "Configure Network Server":
            self.change_nav("Network", "Configure")
        elif selected_option == "Configure Layer Models":
            self.change_nav("Models", "Layer Models")
        elif selected_option == "Configure End Models":
            self.change_nav("Models", "End Models")
        elif selected_option == "Start Job Server":
            if self.provider.job_provider.get_job_port() is None:
                self.provider.job_provider.set_job_port(DEFAULT_JOB_PORT)
            self.provider.job_provider.start_oai_server()
        elif selected_option == "Stop Job Server":
            self.provider.job_provider.stop_oai_server()
        elif selected_option == "Configure Job Server":
            self.change_nav("Jobs", "Server")
        elif selected_option == "Show Logs":
            self.change_nav("Home", "Activity")

    def _get_ram_usage(self) -> List[str]:
        used_ram = self.provider.get_used_system_ram()
        total_ram = self.provider.get_total_system_ram()

        used_swap = self.provider.get_used_swap()
        total_swap = self.provider.get_total_swap()

        lines = [
            f"System RAM:  {used_ram:.1f}/{total_ram:.1f}GB",
            f"System Swap: {used_swap:.1f}/{total_swap:.1f}GB"
        ]

        for i in range(self.provider.get_cuda_device_count()):
            used_vram = self.provider.get_used_cuda_memory(i)
            total_vram = self.provider.get_total_cuda_memory(i)
            lines.append(f"CUDA {i} VRAM: {used_vram:.1f}/{total_vram:.1f}GB")

        return lines

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

        encryption_text = ""
        if self.router_status is not None and self.router_status.encrypted:
            encryption_text = "\nEncrypted connection"

        warning_text = ""
        if self.network_config.node_id is None:
            warning_text = "\nWarning: Node ID must be set to start\nnetwork server, open network config."
        
        return f"{state_label}{peer_text}{encryption_text}{warning_text}"

    def _models_are_starting(self) -> bool:
        for key in self.models_status.keys():
            pipe = self.models_status[key]
            for model in pipe:
                if model.status == ModelStatus.Starting:
                    return True
        return False

    def _get_left_panel(self):
        lines = ["   Options:", ""]
        selected_option = self._get_selected_option()
        for label in self._get_options():
            lines.extend([make_selectable_text(label, label == selected_option), ""])
        lines.extend(["", ""])

        self.models_status = self.provider.model_provider.get_models_status()

        return lines
    
    def _get_network_status(self) -> List[str]:
        lines = []
        if self.router_status is not None or self.network_port_available():
            if self.network_config.node_id != "":
                lines.extend([f"Network: {self._get_network_label()}", ""])
            else:
                lines.extend(["Warning:\nNode ID not set, cannot start network", ""])
        else:
            lines.extend([f"Warning:\nNetwork port {self.network_config.port} not available", ""])
        
        return lines

    def _get_job_status(self):
        lines = ["Job Server:"]

        if self.job_serv_running:
            job_str = f"running on port {self.job_port}"
        elif self.job_port is None:
            job_str = "disabled"
        else:
            job_str = "stopped"
        lines.append(f"Status: {job_str}")

        api_keys = self.provider.job_provider.get_api_keys()
        if len(api_keys) > 0:
            lines.append(f"{len(api_keys)} API Key(s)")

        if self.job_port is not None and not self.job_serv_running and not self.job_port_available():
            lines.append(f"Warning:\nJob port {self.job_port} is not available")

        if self.job_serv_running:
            jobs = self.provider.job_provider.get_active_jobs()
            lines.append(f"{len(jobs)} job(s) running")
            if len(jobs) > 2:
                jobs = jobs[:2]
            for job in jobs:
                token_str = ""
                if job.prompt_processed == 1:
                    token_str = f"{job.current_token} tokens"
                else:
                    token_str = f"{job.prompt_processed * 100:.0f}% processed"
                lines.append(f"ID {job.job_id[:4]} {token_str} ({job.ram:.2f}GB cached)")

        lines.append("")

        return lines
    
    def _get_pipe_status(self, pipe: MetaPipe, num_local_layers: int, local_end_model_ids: Set[str]) -> List[str]:
        entry = []
        entry.extend(format_pipe_view(pipe))
        if pipe.is_complete(num_local_layers) and pipe.model_id in local_end_model_ids:
            entry.append("Ready to serve")
        entry.append("")
        return entry

    def _get_pipes_status(self) -> List[str]:
        lines = []
        self.models_status = self.provider.model_provider.get_models_status()

        local_end_model_ids = {
            model_id for model_id in self.end_models
            if any(s.end_model for s in self.models_status.get(model_id, []))
        }

        lines.append("Connected Pipes:")
        
        connected_pipes = self.provider.pipe_provider.get_connected_pipes()
        if connected_pipes is not None:
            self.connected_pipes = connected_pipes
            if self.connected_pipes is None or len(self.connected_pipes) == 0:
                lines.extend(["None Connected", ""])
            else:
                entries = []
                num_local_layers = ModelProvider.get_num_local_layers()
                for pipe in self.connected_pipes:
                    entries.append(self._get_pipe_status(pipe, num_local_layers, local_end_model_ids))

                lines.extend(make_window_text(entries, self.pipe_idx, 7))
        else:
            lines.append("None Connected")
        
        return lines

    def _get_right_panel(self):
        lines = self._get_ram_usage()
        lines.append("")

        lines.extend(self._get_network_status())
        lines.extend(self._get_job_status())
        lines.extend(self._get_pipes_status())

        return lines

    def get_view(self) -> Tuple[List[str], List[str]]:
        self.layer_models = self.provider.model_provider.get_layer_models()
        self.end_models = self.provider.model_provider.get_end_models()
        self.network_config = self.provider.network_provider.get_network_config()
        self.router_status = self.provider.network_provider.get_network_status()
        self.job_serv_running = self.provider.job_provider.oai_server_running()
        self.job_port = self.provider.job_provider.get_job_port()

        return self._get_left_panel(), self._get_right_panel()

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Esc: Menu"])