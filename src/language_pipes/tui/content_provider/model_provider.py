import io
import os
import toml
import shutil
from enum import Enum
from tqdm.auto import tqdm
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional, Dict
from huggingface_hub import snapshot_download, errors

from language_pipes.modeling.model_manager import ModelManager
from language_pipes.distributed_state_network.util import stop_thread
from language_pipes.util.config import default_model_dir, default_config_dir

class ModelDownloadProgress(tqdm):
    latest_instance: Optional["ModelDownloadProgress"] = None
    _devnull_file: Optional[io.TextIOWrapper] = None

    @classmethod
    def _get_devnull_file(cls) -> io.TextIOWrapper:
        if cls._devnull_file is None or cls._devnull_file.closed:
            cls._devnull_file = open(os.devnull, 'w')
        return cls._devnull_file

    def __init__(self, *args, **kwargs):
        if 'name' in kwargs:
            del kwargs['name']
        # Send fp to devnull so any base-class writes are harmless.
        kwargs['file'] = self._get_devnull_file()
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

class ModelProvider:
    download_model_thread: Optional[Thread]
    download_message: Optional[str]
    downloading_to_folder: Optional[Path]

    def __init__(self, model_manager: ModelManager):
        self.download_model_thread = None
        self.download_message = None
        self.downloading_to_folder = None
    
    # Returns (process_id, status)
    def get_models_status(self) -> Dict[str, ModelStatus]:
        return {}
    
    # Models / Installed
    @staticmethod
    def get_installed_models() -> List[str]:
        models_dir = default_model_dir()

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
        model_dir = Path(default_model_dir()) / model_name
        if not os.path.exists(model_dir):
            return
        shutil.rmtree(model_dir)
        
    def start_download(self, model_id: str):
        if self.download_model_thread is not None:
            return
        clone_dir = Path(default_model_dir()) / model_id / "data"
        self.downloading_to_folder = clone_dir
        self.download_message = None
        def download_model():
            error = False
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=clone_dir,
                    token=self.get_hf_token(),
                    tqdm_class=ModelDownloadProgress
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
    def get_globals() -> Dict:
        global_path = Path(default_config_dir()) / "globals.toml"
        if not os.path.exists(global_path):
            with open(global_path, 'w', encoding="utf-8") as f:
                toml.dump({ }, f)
        
        return toml.loads(global_path.read_text())

    @staticmethod
    def save_globals(data: Dict):
        global_path = Path(default_config_dir()) / "globals.toml"
        if not os.path.exists(global_path):
            return
        with open(global_path, 'w', encoding="utf-8") as f:
            toml.dump(data, f)

    @staticmethod
    def get_hf_token() -> Optional[str]:
        return ModelProvider.get_globals().get("hf_token", None)

    @staticmethod
    def save_hf_token(token: str):
        data = ModelProvider.get_globals()
        data["hf_token"] = token
        ModelProvider.save_globals(data)

    @staticmethod
    def get_models_to_load(config_file: Path) -> List[ModelToLoad]:
        data = toml.loads(config_file.read_text())
        if "models_to_load" not in data:
            return []
        models = []
        for m in data.get("models_to_load", []):
            models.append(ModelToLoad(m.get("model_id", ""), m.get("load_ends", False), m.get("device", ""), m.get("max_memory", "")))
        return models
    
    @staticmethod
    def save_models_to_load(config_file: Path, models: List[ModelToLoad]):
        data = toml.loads(config_file.read_text())
        
        to_load = []
        for m in models:
            to_load.append({
                "model_id": m.model_id,
                "load_ends": m.load_ends,
                "device": m.device,
                "max_memory": m.max_memory
            })

        data["models_to_load"] = to_load

        with open(config_file, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

    @staticmethod
    def validate_device_name(device: str) -> bool:
        import torch
        try:
            torch.device(device)
            return True
        except RuntimeError:
            return False