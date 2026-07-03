import os 
from pathlib import Path

def default_app_dir() -> str:
    return str(Path.home() / ".config" / "language_pipes")

def default_model_dir() -> str:
    return str(Path.home() / ".cache" / "language_pipes" / "models")

def default_log_dir() -> Path:
    return Path.home() / ".cache" / "language_pipes" / "logs"

def get_app_dir() -> Path:
    return Path(os.environ.get("LP_APP_DIR", default_app_dir()))

def get_model_dir() -> Path:
    return Path(os.environ.get("LP_MODEL_DIR", default_model_dir()))

def get_config_files(config_dir: Path):
    if not os.path.exists(config_dir):
        return []
    return [f.replace(".toml", "") for f in os.listdir(config_dir)]

def get_max_node_jobs() -> int:
    return int(os.environ.get("LP_MAX_NODE_JOBS", 10))

def get_max_api_jobs() -> int:
    return int(os.environ.get("LP_MAX_API_JOBS", 5))

def is_8_bit_mode() -> bool:
    return os.environ.get("LP_8_BIT_MODE", "false").lower() in ("1", "true", "yes")