import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import toml
import torch

from distributed_state_network.objects.config import DSNodeConfig
from language_pipes.util.config import get_app_dir

logger = logging.getLogger(__name__)

DEFAULT_NUM_LOCAL_LAYERS = 1
DEFAULT_END_MODEL_DEVICE = "cpu"

def _deprecated_env_num_local_layers() -> Optional[int]:
    raw = os.environ.get("LP_NUM_LOCAL_LAYERS")
    if raw is None:
        return None
    logger.warning(
        "LP_NUM_LOCAL_LAYERS is deprecated; set num_local_layers per end model "
        "in the end_models configuration instead."
    )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

@dataclass
class EndModelConfig:
    model_id: str
    num_local_layers: int = DEFAULT_NUM_LOCAL_LAYERS
    # The PyTorch device used for both the local layers and the embed/head
    # modules of this end model.
    device: str = DEFAULT_END_MODEL_DEVICE

    def _has_only_defaults(self) -> bool:
        return (
            self.num_local_layers == DEFAULT_NUM_LOCAL_LAYERS
            and self.device == DEFAULT_END_MODEL_DEVICE
        )

    def to_config(self) -> Union[str, Dict[str, Any]]:
        # Preserve the simple string form when no extra options are set.
        if self._has_only_defaults():
            return self.model_id
        return {
            "model_id": self.model_id,
            "num_local_layers": self.num_local_layers,
            "device": self.device,
        }

    @staticmethod
    def from_config(data: Union[str, Dict[str, Any]]) -> "EndModelConfig":
        default = _deprecated_env_num_local_layers()
        if default is None:
            default = DEFAULT_NUM_LOCAL_LAYERS

        if isinstance(data, str):
            return EndModelConfig(model_id=data, num_local_layers=default)

        return EndModelConfig(
            model_id=data.get("model_id", ""),
            num_local_layers=data.get("num_local_layers", default),
            device=data.get("device", DEFAULT_END_MODEL_DEVICE),
        )

def _serialize_end_models(
    end_models: List["EndModelConfig"],
) -> Union[List[str], List[Dict[str, Any]]]:
    """Serialize end models to a TOML-friendly, homogeneous list.

    TOML arrays cannot mix scalars and tables, so the whole list is written in
    a single form: a bare string array when every model uses the default number
    of local layers, otherwise an array of tables.
    """
    if all(m._has_only_defaults() for m in end_models):
        return [m.model_id for m in end_models]
    return [
        {
            "model_id": m.model_id,
            "num_local_layers": m.num_local_layers,
            "device": m.device,
        }
        for m in end_models
    ]

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
    job_port: Optional[int]
    api_keys: List[str]
    layer_models: List[ModelToLoad]
    end_models: List[EndModelConfig]

    network_config: DSNodeConfig

    _file_path: Optional[Path]

    def __init__(self):
        self.job_port = None
        self.api_keys = []
        self.layer_models = []
        self.end_models = []
        self._file_path = None
        self.network_config = DSNodeConfig.from_dict({ })
    
    def save(self):
        if self._file_path is None:
            return
        
        data = {
            "api_keys": self.api_keys,
            "layer_models": [o.to_dict() for o in self.layer_models],
            "end_models": _serialize_end_models(self.end_models),
            "node_id": self.network_config.node_id,
            "peer_port": self.network_config.port,
            "network_ip": self.network_config.network_ip,
            "network_key": self.network_config.aes_key,
            "whitelist_node_ids": self.network_config.whitelist_node_ids,
            "bootstrap_nodes": [{
                "address": o.address,
                "port": o.port
            } for o in self.network_config.bootstrap_nodes]
        }
        if self.job_port is not None:
            data["job_port"] = self.job_port

        with open(self._file_path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

    def to_string(self) -> str:
        lines = [
            "=" * 60,
            "--- Configuration Settings ---",
            "=" * 60,
            "",
            f"Job Port: {self.job_port if self.job_port is not None else 'Disabled'}"
        ]

        lines.append("API Keys:")
        if len(self.api_keys) > 0:
            for key in self.api_keys:
                lines.append(f"- {key}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("Layer Models:")
        if len(self.layer_models) > 0:
            for model in self.layer_models:
                lines.extend([
                    f"Model ID: {model.model_id}",
                    f"Max Memory: {model.memory}",
                    f"Device: {model.device}",
                    ""
                ])
        else:
            lines.append("- None")
        
        lines.append("")
        lines.append("End Models:")
        if len(self.end_models) > 0:
            for model in self.end_models:
                lines.append(f"- {model.model_id} (local layers: {model.num_local_layers}, device: {model.device})")
        else:
            lines.append("- None")
        
        lines.append(self.network_config.to_string())

        return "\n".join(lines)

    @staticmethod
    def from_file(file_path: Path) -> 'LpConfig':
        cfg = LpConfig()
        
        if not os.path.exists(file_path):
            return cfg

        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)

        cfg.job_port = data.get("job_port")
        cfg.api_keys = data.get("api_keys", [])
        cfg.layer_models = [ModelToLoad.from_dict(o) for o in data.get("layer_models", [])]
        cfg.end_models = [EndModelConfig.from_config(o) for o in data.get("end_models", [])]
        cfg.network_config = DSNodeConfig.from_dict({
            "credential_dir": str(get_app_dir() / "credentials"),
            "logging_dir": str(get_app_dir() / "logs"),
            "node_id": data.get("node_id", None),
            "aes_key": data.get("network_key", None),
            "network_ip": data.get("network_ip", None),
            "port": data.get("peer_port", 5000),
            "bootstrap_nodes": data.get("bootstrap_nodes", []),
            "whitelist_node_ids": data.get("whitelist_node_ids", [])
        })
        
        cfg._file_path = file_path

        return cfg
