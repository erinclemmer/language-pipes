import io
import os
import shutil
from enum import Enum
import torch
from tqdm.auto import tqdm
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Tuple
from huggingface_hub import snapshot_download, errors

from language_pipes.config import LpConfig, ModelToLoad
from language_pipes.global_config import GlobalConfig
from language_pipes.modeling.end_model import EndModel
from language_pipes.modeling.llm_meta_data import LlmMetadata
from language_pipes.modeling.llm_model import LlmModel
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.pipes.router_pipes import RouterPipes
from distributed_state_network.util import stop_thread
from language_pipes.util.config import get_model_dir


class ModelDownloadProgress(tqdm):
    latest_instance: Optional["ModelDownloadProgress"] = None
    _devnull_file: Optional[io.TextIOWrapper] = None

    @classmethod
    def _get_devnull_file(cls) -> io.TextIOWrapper:
        if cls._devnull_file is None or cls._devnull_file.closed:
            cls._devnull_file = open(os.devnull, "w")
        return cls._devnull_file

    def __init__(self, *args, **kwargs):
        if "name" in kwargs:
            del kwargs["name"]
        # Send fp to devnull so any base-class writes are harmless.
        kwargs["file"] = self._get_devnull_file()
        super().__init__(*args, **kwargs)
        ModelDownloadProgress.latest_instance = self

    # -- output suppression / capture ------------------------------------------

    def display(self, msg=None, pos=None):  # type: ignore[override]
        pass

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        pass

    def clear(self, *args, **kwargs):
        pass

    def close(self):
        if self.disable:
            return
        self.disable = True
        lock = self.get_lock()
        with lock:
            type(self)._instances.discard(self)  # type: ignore[attr-defined]


class ModelStatus(Enum):
    Stopped = "Stopped"
    Starting = "Starting"
    Running = "Running"
    Stopping = "Stopping"


@dataclass
class ModelStatusInfo:
    status: ModelStatus
    device: torch.device
    pipe_id: str
    start_layer: int
    end_layer: int
    num_layers: int
    end_model: bool
    ram_used: float

class ModelProvider:
    download_model_thread: Optional[Thread]
    download_message: Optional[str]
    downloading_to_folder: Optional[Path]
    config_file: Path

    get_router_pipes: Callable[[], Optional[RouterPipes]]
    get_model_manager: Callable[[], ModelManager]

    def __init__(self, config_file: Path, get_model_manager: Callable, get_router_pipes: Callable):
        self.download_model_thread = None
        self.download_message = None
        self.downloading_to_folder = None
        self.config_file = config_file
        self.get_model_manager = get_model_manager
        self.get_router_pipes = get_router_pipes
        
    def get_model_manager_logs(self) -> List[Tuple[float, str]]:
        mm = self.get_model_manager()
        return mm.logs

    def reset_model_manager_logs(self):
        self.get_model_manager().logs = []

    # Returns a mapping of model_id -> list of lifecycle statuses based on ModelManager state.
    def get_models_status(self) -> Dict[str, List[ModelStatusInfo]]:
        status_by_model: Dict[str, List[ModelStatusInfo]] = {}

        # Initialize with empty lists for all known model_ids
        known_model_ids = set(self.get_model_manager().pipes_hosted.keys())
        known_model_ids.update(
            model.model_id for model in self.get_model_manager().layer_models
        )
        known_model_ids.update(
            model.model_id for model in self.get_model_manager().end_models
        )

        for model_id in known_model_ids:
            status_by_model[model_id] = []

        mm: ModelManager = self.get_model_manager()
        # Add status for each model instance
        for model in mm.layer_models:
            model: LlmModel = model
            status = ModelStatus.Running if model.loaded else ModelStatus.Starting
            status_by_model[model.model_id].append(
                ModelStatusInfo(
                    status=status, 
                    device=model.device,
                    start_layer=model.start_layer, 
                    end_layer=model.end_layer, 
                    end_model=False,
                    num_layers=model.num_hidden_layers,
                    pipe_id=model.pipe_id,
                    ram_used=model.ram_used
                )
            )

        # Add status for end models
        for end_model in mm.end_models:
            is_loaded = end_model.loaded
            status = ModelStatus.Running if is_loaded else ModelStatus.Starting
            status_by_model[end_model.model_id].append(
                ModelStatusInfo(status=status, device=end_model.device, start_layer=-1, end_layer=-1, end_model=True, num_layers=0, pipe_id='', ram_used=0)
            )

        return status_by_model

    @staticmethod
    def get_model_metadata(model_id: str) -> LlmMetadata:
        return LlmMetadata(get_model_dir() / model_id)
        
    # Models / Installed
    @staticmethod
    def get_installed_models() -> List[str]:
        models_dir = get_model_dir()

        models = []
        if not os.path.exists(models_dir):
            return models

        for org in os.listdir(models_dir):
            org_path = os.path.join(models_dir, org)
            if os.path.isdir(org_path):
                for model in os.listdir(org_path):
                    model_path = os.path.join(org_path, model)
                    if os.path.isdir(model_path):
                        models.append(f"{org}/{model}")

        return sorted(models, key=lambda m: m.lower())

    @staticmethod
    def delete_installed_model(model_name: str):
        model_dir = get_model_dir() / model_name
        if not os.path.exists(model_dir):
            return
        shutil.rmtree(model_dir)

    def start_download(self, model_id: str, token: Optional[str] = None):
        if self.download_model_thread is not None:
            return
        clone_dir = get_model_dir() / model_id / "data"
        self.downloading_to_folder = clone_dir
        self.download_message = None

        def download_model():
            error = False
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=clone_dir,
                    token=token,
                    tqdm_class=ModelDownloadProgress,
                )
            except errors.RepositoryNotFoundError:
                self.download_message = "[ERROR] Repository not found"
                error = True
            except errors.HFValidationError:
                self.download_message = "[ERROR] Invalid repository ID"
                error = True
            except errors.LocalEntryNotFoundError:
                self.download_message = "[Error] Connection to huggingface server failed"
            except RuntimeError:
                self.download_message = "[ERROR] Download stopped"
                error = True
            self.download_model_thread = None
            self.downloading_to_folder = None
            if error:
                shutil.rmtree(get_model_dir() / model_id)
            else:
                # Compile metadata
                self.download_message = "Computing metadata..."
                ModelProvider.get_model_metadata(model_id)
                self.download_message = "[SUCCESS] Download complete"

        self.download_model_thread = Thread(target=download_model, args=())
        self.download_model_thread.start()

    def stop_model_download(self):
        if self.download_model_thread is None:
            return
        stop_thread(self.download_model_thread)
        shutil.rmtree(str(self.downloading_to_folder))
        self.download_model_thread = None

    def check_download_progress(self) -> Optional[str]:
        if self.download_message is not None:
            return self.download_message
        if self.download_model_thread is None:
            return None
        if ModelDownloadProgress.latest_instance is None:
            return None
        return str(ModelDownloadProgress.latest_instance)

    def load_layer_model(self, model: ModelToLoad):
        rp = self.get_router_pipes()
        if rp is None:
            return

        def host_layer_model():
            self.get_model_manager().host_model(
                node_id=rp.router.node_id(),
                router_pipes=rp,
                model_id=model.model_id,
                max_memory=model.memory,
                device=model.device,
                first_layer=0,
            )

        Thread(target=host_layer_model, args=()).start()

    def restart_layer_model(self, old_model: ModelToLoad, new_model: ModelToLoad):
        rp = self.get_router_pipes()
        if rp is None:
            return

        def restart_model():
            model_manager = self.get_model_manager()
            model_manager.shutdown_layer_models(rp, old_model.model_id, old_model.device)
            model_manager.host_model(
                node_id=rp.router.node_id(),
                router_pipes=rp,
                model_id=new_model.model_id,
                max_memory=new_model.memory,
                device=new_model.device,
                first_layer=0,
            )

        Thread(target=restart_model, args=()).start()

    def load_end_model(self, model_id: str):
        def host_end_model():
            self.get_model_manager().load_end_model(model_id, "cpu", self.get_num_local_layers())

        Thread(target=host_end_model, args=()).start()

    @staticmethod
    def get_num_local_layers():
            return EndModel.get_num_local_layers()

    def unload_layer_models(self, model_id: str, device: torch.device):
        rp = self.get_router_pipes()
        if rp is None:
            return
        def shutdown_layer_models():
            self.get_model_manager().shutdown_layer_models(rp, model_id, device)
        Thread(target=shutdown_layer_models, args=()).start()

    def unload_all_models(self):
        mm = self.get_model_manager()
        for m in mm.layer_models:
            self.unload_layer_models(m.model_id, m.device)

        for m in mm.end_models:
            self.unload_end_model(m.model_id)

    def unload_end_model(self, model_id: str):
        def shutdown_end_model():
            self.get_model_manager().shutdown_end_model(model_id)
        Thread(target=shutdown_end_model, args=()).start()

    def get_layer_models(self) -> List[ModelToLoad]:
        cfg = LpConfig.from_file(self.config_file)
        return sorted(cfg.layer_models, key=lambda m: m.model_id.lower())
        
    def save_layer_models(self, models: List[ModelToLoad]):
        cfg = LpConfig.from_file(self.config_file)
        cfg.layer_models = models
        cfg.save()

    def get_end_models(self) -> List[str]:
        return sorted(LpConfig.from_file(self.config_file).end_models, key=lambda m:m.lower())
        
    def save_end_models(self, end_models: List[str]):
        cfg = LpConfig.from_file(self.config_file)
        cfg.end_models = end_models
        cfg.save()

    @staticmethod
    def validate_device_name(device: str) -> bool:
        try:
            import torch
            torch.device(device) # type: ignore
            return True
        except RuntimeError:
            return False

    @staticmethod
    def get_hf_config_token() -> Optional[str]:
        cfg = GlobalConfig.from_file()
        return cfg.hf_token

    @staticmethod
    def save_hf_token(token: str):
        cfg = GlobalConfig.from_file()
        cfg.hf_token = token
        cfg.save()
