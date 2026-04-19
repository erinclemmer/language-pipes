import os
from typing import Dict, Optional

def get_env_config() -> Dict[str, Optional[str]]:
    return {
        "app_dir": os.getenv("LP_APP_DIR"),
        "model_dir": os.getenv("LP_MODEL_DIR"),
        "huggingface_token": os.getenv("LP_HUGGINGFACE_TOKEN"),
    }