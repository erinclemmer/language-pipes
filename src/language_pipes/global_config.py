import os
from pathlib import Path
from typing import Optional

import toml

from language_pipes.util.config import get_app_dir

class GlobalConfig:
    hf_token: Optional[str]

    _file_path: Optional[Path]

    def __init__(self):
        self.hf_token = None
        self._file_path = None

    def save(self):
        if self._file_path is None:
            return
        
        with open(self._file_path, 'w', encoding='utf-8') as f:
            toml.dump({
                "hf_token": self.hf_token
            }, f)

    @staticmethod
    def from_file() -> 'GlobalConfig':
        cfg = GlobalConfig()
        
        file_path = get_app_dir() / "globals.toml"
        
        if not os.path.exists(file_path):
            cfg._file_path = file_path
            return cfg

        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        
        cfg.hf_token = data.get('hf_token', None)

        cfg._file_path = file_path

        return cfg