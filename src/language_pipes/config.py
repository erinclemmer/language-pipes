import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import toml
import torch

from language_pipes.distributed_state_network.objects.config import DSNodeConfig

@dataclass
class ModelToLoad:
    model_id: str
    device: torch.device
    memory: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "memory": self.memory
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return ModelToLoad(
            model_id=data.get("model_id", ""),
            device=torch.device(data.get("device", "cpu")),
            memory=data.get("memory", 0)
        )

class LpConfig:
    oai_port: int
    api_keys: List[str]
    layer_models: List[ModelToLoad]
    end_models: List[str]

    network_config: DSNodeConfig

    _file_path: Optional[Path]

    def __init__(self):
        self.oai_port = 8000 
        self.api_keys = []
        self.layer_models = []
        self._file_path = None
        self.network_config = DSNodeConfig.from_dict({ })
    
    def save(self):
        if self._file_path is None:
            return
        
        with open(self._file_path, 'w', encoding='utf-8') as f:
            toml.dump({
                "oai_port": self.oai_port,
                "api_keys": self.api_keys,
                "layer_models": [o.to_dict() for o in self.layer_models],
                "end_models": self.end_models,
                "network_config": self.network_config.to_dict()
            }, f)

    def apply_overrides(self, data: Dict[str, Any]):
        if "oai_port" in data:
            self.oai_port = data["oai_port"]


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
        cfg.end_models = data.get("end_models", [])
        cfg.network_config = DSNodeConfig.from_dict(data.get("network_config", { }))
        
        cfg._file_path = file_path

        return cfg