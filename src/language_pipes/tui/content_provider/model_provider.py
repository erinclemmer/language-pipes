import io
import os
import toml
import shutil
from enum import Enum
import torch
from tqdm.auto import tqdm
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Tuple
from huggingface_hub import snapshot_download, errors

from language_pipes.global_config import GlobalConfig
from language_pipes.modeling.llm_model import LlmModel
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.pipes.router_pipes import RouterPipes
from language_pipes.distributed_state_network.util import stop_thread
from language_pipes.util.config import get_model_dir, get_app_dir


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


@dataclass
class ModelToLoad:
    model_id: str
    load_ends: bool
    device: str
    max_memory: float


class ModelStatus(Enum):
    Stopped = "Stopped"
    Starting = "Starting"
    Running = "Running"
    Stopping = "Stopping"


@dataclass
class ModelStatusInfo:
    status: ModelStatus
    pipe_id: str
    start_layer: int
    end_layer: int
    num_layers: int
    end_model: bool

class ModelProvider:
    download_model_thread: Optional[Thread]
    download_message: Optional[str]
    downloading_to_folder: Optional[Path]

    get_router_pipes: Callable[[], Optional[RouterPipes]]

    def __init__(self, get_model_manager: Callable, get_router_pipes: Callable):
        self.download_model_thread = None
        self.download_message = None
        self.downloading_to_folder = None
        self.get_model_manager = get_model_manager
        self.get_router_pipes = get_router_pipes

    def get_model_manager_logs(self) -> List[Tuple[float, str]]:
        mm = self.get_model_manager()
        return mm.logs

    # Returns a mapping of model_id -> list of lifecycle statuses based on ModelManager state.
    def get_models_status(self) -> Dict[str, List[ModelStatusInfo]]:
        status_by_model: Dict[str, List[ModelStatusInfo]] = {}

        # Initialize with empty lists for all known model_ids
        known_model_ids = set(self.get_model_manager().pipes_hosted.keys())
        known_model_ids.update(
            model.model_id for model in self.get_model_manager().models
        )
        known_model_ids.update(
            model.model_id for model in self.get_model_manager().end_models
        )

        for model_id in known_model_ids:
            status_by_model[model_id] = []

        mm: ModelManager = self.get_model_manager()
        # Add status for each model instance
        for model in mm.models:
            model: LlmModel = model
            status = ModelStatus.Running if model.loaded else ModelStatus.Starting
            status_by_model[model.model_id].append(
                ModelStatusInfo(
                    status=status, 
                    start_layer=model.start_layer, 
                    end_layer=model.end_layer, 
                    end_model=False,
                    num_layers=model.num_hidden_layers,
                    pipe_id=model.pipe_id
                )
            )

        # Add status for end models
        for end_model in mm.end_models:
            is_loaded = getattr(end_model, "loaded", True)
            status = ModelStatus.Running if is_loaded else ModelStatus.Starting
            status_by_model[end_model.model_id].append(
                ModelStatusInfo(status=status, start_layer=-1, end_layer=-1, end_model=True, num_layers=0, pipe_id='')
            )

        return status_by_model

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

        return sorted(models)

    @staticmethod
    def delete_installed_model(model_name: str):
        model_dir = get_model_dir() / model_name
        if not os.path.exists(model_dir):
            return
        shutil.rmtree(model_dir)

    def start_download(self, model_id: str):
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
                    token=self.get_hf_token(),
                    tqdm_class=ModelDownloadProgress,
                )
            except errors.RepositoryNotFoundError:
                self.download_message = "[ERROR] Repository not found"
                error = True
            except errors.HFValidationError:
                self.download_message = "[ERROR] Invalid repository ID"
                error = True
            except RuntimeError:
                self.download_message = "[ERROR] Download stopped"
                error = True
            self.download_model_thread = None
            self.downloading_to_folder = None
            if not error:
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

    @staticmethod
    def get_hf_token() -> Optional[str]:
        cfg = GlobalConfig.from_file()
        return cfg.hf_token

    @staticmethod
    def save_hf_token(token: str):
        cfg = GlobalConfig.from_file()
        return cfg

    def host_model(self, model: ModelToLoad):
        rp = self.get_router_pipes()
        if rp is None:
            return

        def host():
            mm: ModelManager = self.get_model_manager()
            mm.host_model(
                node_id=rp.router.node_id(),
                router_pipes=rp,
                model_id=model.model_id,
                max_memory=model.max_memory,
                device=torch.device(model.device),
                first_layer=0,
            )
            if model.load_ends:
                mm.load_end_model(model.model_id, "cpu", 0)

        Thread(target=host, args=()).start()

    def shutdown_models(self, model_id: str):
        rp = self.get_router_pipes()
        if rp is None:
            return
        def shutdown():
            self.get_model_manager().shutdown_models(rp, model_id)
        Thread(target=shutdown, args=()).start()

    @staticmethod
    def get_models_to_load(config_file: Path) -> List[ModelToLoad]:
        data = toml.loads(config_file.read_text())
        if "models_to_load" not in data:
            return []
        models = []
        for m in data.get("models_to_load", []):
            models.append(
                ModelToLoad(
                    m.get("model_id", ""),
                    m.get("load_ends", False),
                    m.get("device", ""),
                    m.get("max_memory", ""),
                )
            )
        return models

    @staticmethod
    def save_models_to_load(config_file: Path, models: List[ModelToLoad]):
        data = toml.loads(config_file.read_text())

        to_load = []
        for m in models:
            to_load.append(
                {
                    "model_id": m.model_id,
                    "load_ends": m.load_ends,
                    "device": m.device,
                    "max_memory": m.max_memory,
                }
            )

        data["models_to_load"] = to_load

        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(data, f)

    @staticmethod
    def validate_device_name(device: str) -> bool:
        import torch

        try:
            torch.device(device)
            return True
        except RuntimeError:
            return False
