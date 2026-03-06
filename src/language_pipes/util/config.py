import os 
from pathlib import Path

def default_config_dir() -> str:
    return str(Path.home() / ".config" / "language_pipes")


def default_model_dir() -> str:
    return str(Path.home() / ".cache" / "language_pipes" / "models")

def get_config_files(config_dir: str):
    if not os.path.exists(config_dir):
        return []
    return [f.replace(".toml", "") for f in os.listdir(config_dir)]