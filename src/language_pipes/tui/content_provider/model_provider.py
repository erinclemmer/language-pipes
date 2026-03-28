import io
import os
import shutil
from tqdm.auto import tqdm
from pathlib import Path
from threading import Thread
from typing import List, Optional
from huggingface_hub import snapshot_download, errors

from language_pipes.distributed_state_network.util import stop_thread
from language_pipes.util.config import default_model_dir

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




class ModelProvider:
    download_model_thread: Optional[Thread]
    download_message: Optional[str]
    downloading_to_folder: Optional[Path]

    def __init__(self):
        self.download_model_thread = None
        self.download_message = None
        self.downloading_to_folder = None
    
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
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=clone_dir,
                    token=None,
                    tqdm_class=ModelDownloadProgress
                )
            except errors.RepositoryNotFoundError:
                self.download_message = "[ERROR] Repository not found"
            except errors.HFValidationError:
                self.download_message = "[ERROR] Invalid repository ID"
            except RuntimeError:
                self.download_message = "[ERROR] Download stopped"
            self.download_model_thread = None
            self.downloading_to_folder = None
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