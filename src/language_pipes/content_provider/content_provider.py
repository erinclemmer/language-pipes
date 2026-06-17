from dataclasses import dataclass
import os
from pathlib import Path
from time import sleep

import psutil
import torch
from typing import Callable, List, Optional, Dict

from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_receiver import JobReceiver
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.util.byte_helper import ByteHelper
from language_pipes.util.config import get_model_dir
from language_pipes.util.utils import is_port_available
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.content_provider.job_provider import JobProvider
from language_pipes.distributed_state_network.handler import DSNodeServer
from language_pipes.content_provider.pipe_provider import PipeProvider
from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.content_provider.network_provider import NetworkProvider

@dataclass
class ProviderState:
    visible_headers: List[str]
    visible_sub_menu: Dict[str, List[str]]

class ContentProvider:
    router: Optional[DSNodeServer]
    router_pipes: Optional[RouterPipes]
    pipe_manager: Optional[PipeManager]
    job_tracker: Optional[JobTracker]
    job_factory: Optional[JobFactory]
    job_receiver: Optional[JobReceiver]

    model_manager: ModelManager
    model_provider: ModelProvider
    network_provider: NetworkProvider
    pipe_provider: PipeProvider
    job_provider: JobProvider
    config_file: Path
    create_alert: Callable[[str], None]

    def __init__(self, config_file: Path, create_alert: Callable[[str], None]):
        self.router = None
        self.router_pipes = None
        self.pipe_manager = None
        self.job_tracker = None
        self.job_factory = None
        self.job_receiver = None
        self.model_manager = ModelManager()
        self.config_file = config_file
        self.create_alert = create_alert
        self.state = ProviderState([], {})

        self.model_provider = ModelProvider(config_file, lambda: self.model_manager, lambda: self.router_pipes)
        self.network_provider = NetworkProvider(config_file, lambda: self.router, self.set_router, self.create_alert)
        self.pipe_provider = PipeProvider(lambda: self.pipe_manager)
        self.job_provider = JobProvider(
            config_file, 
            lambda: self.router_pipes, 
            lambda: self.model_manager, 
            lambda: self.pipe_manager,
            lambda: self.job_tracker,
            lambda: self.job_factory,
            lambda: self.job_receiver
        )
        self.sync_provider_state()

    def sync_provider_state(self):
        self.state.visible_headers = ["Home", "Network", "Models"]
        if self.router is not None and self.router.running:
            self.state.visible_headers.extend(["Pipes", "Jobs"])

        self.state.visible_sub_menu = {
            "Home": ["Dashboard", "Activity"]
        }

        network_paths = []
        network_config = self.network_provider.get_network_config()
        if network_config.node_id is not None:
            network_paths.append("Status")
        
        if self.router is not None and self.router.running:
            network_paths.append("Peers")
        
        network_paths.append("Configure")

        self.state.visible_sub_menu["Network"] = network_paths

        model_paths = []

        if self.router is not None and self.router.running:
            model_paths.extend(["Layer Models", "End Models"])

        model_paths.append("Installed")

        self.state.visible_sub_menu["Models"] = model_paths

        if "Pipes" in self.state.visible_headers:
            self.state.visible_sub_menu["Pipes"] = ["Connected", "Complete", "Incomplete"]

        if "Jobs" in self.state.visible_headers:
            self.state.visible_sub_menu["Jobs"] = ["Server", "Active"]

    def set_router(self, router: Optional[DSNodeServer]):
        self.router = router
        if router is not None:
            self.router_pipes = RouterPipes(router)
            self.pipe_manager = PipeManager(self.model_manager, self.router_pipes)
            self.job_tracker = JobTracker()
            self.job_factory = JobFactory(self.job_tracker, self.pipe_manager)
            self.job_receiver = JobReceiver(
                job_factory=self.job_factory,
                job_tracker=self.job_tracker,
                model_manager=self.model_manager,
                pipe_manager=self.pipe_manager,
                is_shutdown=self.router_pipes.router.is_shut_down
            )

            self.router_pipes.router.set_receive_cb(self._receive_data)
        else:
            self.router_pipes = None
            self.pipe_manager = None

    def _receive_data(self, data: bytes):
        bts = ByteHelper(data)
        protocol = bts.read_int() # Protocol number
        if protocol == 0 and self.job_receiver is not None:
            self.job_receiver.receive_data(bts.read_bytes())
        if protocol == 1:
            self._handle_rfm(data)

    def _handle_rfm(self, data: bytes):
        bts = ByteHelper(data)
        bts.read_int() # Skip protocol
        req_type = bts.read_int()
        if req_type == 0: # Who has model
            installed_models = self.model_provider.get_installed_models()
            requesting_node = bts.read_string()
            requested_model = bts.read_string()
            if requested_model in installed_models:
                 self._send_rfm_have_model(requesting_node, requested_model)
        if req_type == 1: # I have model
            self._receive_rfm_have_model(data)
        if req_type == 2: # Ready to receive
            installed_models = self.model_provider.get_installed_models()
            requesting_node = bts.read_string()
            requested_model = bts.read_string()
            if requested_model in installed_models:
                self._send_rfm_files(requesting_node, requested_model)
        if req_type == 3: # I'm sending data
            self._receive_rfm_data(data)

    def _receive_rfm_have_model(self, data: bytes):
        if self.model_provider.rfm_model is None or self.model_provider.rfm_node is not None:
            return
        
        bts = ByteHelper(data)
        bts.read_int() # Skip protocol
        bts.read_int() # Skip type
        self.model_provider.rfm_node = bts.read_string()

        bts = ByteHelper()
        bts.write_int(1) # RFM Protocol
        bts.write_int(2) # Ready to receive
        assert self.router is not None
        bts.write_string(self.router.node_id())
        bts.write_string(self.model_provider.rfm_model)
        self.router.send_to_node(self.model_provider.rfm_node, bts.get_bytes())

    def _send_rfm_have_model(self, node_id: str, model_id: str):
        assert self.router is not None

        bts = ByteHelper()
        bts.write_int(1) # RFM Protocol
        bts.write_int(1) # I have model
        bts.write_string(self.router.node_id())
        self.router.send_to_node(node_id, bts.get_bytes())

    def _send_rfm_files(self, node_id: str, model_id: str):
        model_dir = get_model_dir() / model_id / "data"
        assert os.path.exists(model_dir)
        for file in os.listdir(model_dir):
            self._send_rfm_file(node_id, model_id, file, model_dir / file)

    def _send_rfm_file(self, node_id: str, model_id: str, file_name: str, file_path: Path):
        with open(file_path, 'rb') as f:
            idx = 0
            while True:
                pkt_data = f.read(64 * 1024 * 1024) # Read up to 64MB
                if len(pkt_data) == 0:
                    self._send_rfm_packet(node_id, model_id, file_name, idx, None)
                    break
                else:
                    self._send_rfm_packet(node_id, model_id, file_name, idx, pkt_data)

    def _send_rfm_packet(self, node_id: str, model_id: str, file_name: str, idx: int, data: Optional[bytes]):
        bts = ByteHelper()
        bts.write_int(1) # RFM protocol
        bts.write_int(3) # I'm sending data
        bts.write_string(model_id)
        bts.write_string(file_name)
        bts.write_int(idx) # Packet index for file
        if data is None:
            bts.write_int(1) # File done
            bts.write_bytes(b'')
        else:
            bts.write_int(0) # More file data
            bts.write_bytes(data) # File snippet

        assert self.router is not None
        self.router.send_to_node(node_id, bts.get_bytes())

    def _receive_rfm_data(self, data: bytes):
        bts = ByteHelper(data)
        bts.read_int() # Skip Protocol
        bts.read_int() # Skip type
        model_id = bts.read_string()
        if self.model_provider.rfm_model != model_id:
            return
        file_name = bts.read_string()
        packet_idx = bts.read_int()
        file_done = bts.read_int() == 1
        packet_data = bts.read_bytes()

        if self.model_provider.rfm_file_data is None:
            self.model_provider.rfm_file_data = { }

        if file_name not in self.model_provider.rfm_file_data:
            self.model_provider.rfm_file_data[file_name] = { }
        
        if not file_done:
            self.model_provider.rfm_file_data[file_name][str(packet_idx)] = packet_data
        else:
            self.model_provider.rfm_file_data[file_name][str(packet_idx)] = b'EOF'

        # Race condition may make it neccessary to sleep here before testing if everything has made it
        if self._rfm_is_file_done(file_name):
            self._rfm_write_file(model_id, file_name)
            del self.model_provider.rfm_file_data[file_name]

    def _rfm_write_file(self, model_id: str, file_name: str):
        assert ".." not in model_id # Prevent arbitrary writes
        assert self.model_provider.rfm_file_data is not None 
        assert file_name in self.model_provider.rfm_file_data
        
        model_dir = get_model_dir() / model_id / "data"
        if not os.path.exists(model_dir):
            model_dir.mkdir(parents=True)

        with open(model_dir / file_name, "wb") as f:
            idx = 0
            file_data = self.model_provider.rfm_file_data[file_name]
            while True:
                assert str(idx) in file_data
                
                file_packet = file_data[str(idx)]
                if file_packet == b'EOF':
                    break
                f.write(file_packet)
                idx += 1

    def _rfm_is_file_done(self, file_name: str) -> bool:
        if self.model_provider.rfm_file_data is None or file_name not in self.model_provider.rfm_file_data:
            return False
        file_data = self.model_provider.rfm_file_data[file_name]
        
        idx = 0
        while True:
            if str(idx) in file_data:
                pkt_data = file_data[str(idx)]
                if pkt_data == b'EOF':
                    return True
            else:
                return False
            idx += 1

    def stop_network(self):
        if self.router is None:
            return
        self.model_provider.unload_all_models()
        self.job_provider.stop_oai_server()
        self.network_provider._stop_network()
        if self.job_tracker is not None:
            self.job_tracker.shutdown = True
        if self.job_receiver is not None:
            self.job_receiver.shutdown = True
        

    @staticmethod
    def get_total_system_ram() -> float:
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_used_system_ram() -> float:
        return psutil.virtual_memory().used / (1024**3)
    
    @staticmethod
    def get_total_swap() -> float:
        return psutil.swap_memory().total / (1024**3)
    
    @staticmethod
    def get_used_swap() -> float:
        return psutil.swap_memory().used / (1024**3)

    @staticmethod
    def get_cuda_device_count() -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    @staticmethod
    def get_total_cuda_memory(device: int) -> float:
        _, total = torch.cuda.mem_get_info(device)
        return total / (1024**3)

    @staticmethod
    def get_used_cuda_memory(device: int) -> float:
        free, total = torch.cuda.mem_get_info(torch.device(f'cuda:{device}'))
        return (total - free) / (1024**3)

    @staticmethod
    def get_ram_usage() -> str:
        used_ram = ContentProvider.get_used_system_ram()
        total_ram = ContentProvider.get_total_system_ram()

        used_swap = ContentProvider.get_used_swap()
        total_swap = ContentProvider.get_total_swap()
        
        return f"System RAM:  {used_ram:.1f}/{total_ram:.1f}GB".ljust(26) + f"System Swap: {used_swap:.1f}/{total_swap:.1f}GB"

    @staticmethod
    def is_port_available(port: int) -> bool:
        return is_port_available(port)
    
    def shutdown(self):
        if self.router is not None:
            self.stop_network()
            self.set_router(None)
        
        self.model_manager.stop()
        self.job_provider.stop_oai_server()