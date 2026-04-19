import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import toml
import torch

@dataclass
class ModelToLoad:
    model_id: str
    load_ends: bool
    device: torch.device
    memory: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "load_ends": self.load_ends,
            "device": str(self.device),
            "memory": self.memory
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return ModelToLoad(
            model_id=data.get("model_id", ""),
            load_ends=data.get("load_ends", False),
            device=torch.device(data.get("device", "cpu")),
            memory=data.get("memory", 0)
        )

class LpConfig:
    oai_port: int
    api_keys: List[str]
    layer_models: List[ModelToLoad]

    _file_path: Optional[Path]

    def __init__(self):
        self.oai_port = 8000 
        self.api_keys = []
        self.layer_models = []
        self._file_path = None
    
    def save(self):
        if self._file_path is None:
            return
        
        with open(self._file_path, 'w', encoding='utf-8') as f:
            toml.dump({
                "oai_port": self.oai_port,
                "api_keys": self.api_keys,
                "layer_models": [o.to_dict() for o in self.layer_models]
            }, f)

    @staticmethod
    def from_file(file_path: Path) -> 'LpConfig':
        cfg = LpConfig()
        
        if not os.path.exists(file_path):
            return cfg

        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)

        cfg.oai_port = data.get("oai_port", 8000)
        cfg.api_keys = data.get("api_keys", [])
        cfg.layer_models = [ModelToLoad.from_dict(o) for o in data.get("layer_models", [])]
        
        cfg._file_path = file_path

        return cfg
        