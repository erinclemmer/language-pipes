import os
import json
from pathlib import Path
import torch
from typing import List, Tuple, Optional

from language_pipes.llm_layer_collector.auto.auto_rms import AutoRMSNorm
from language_pipes.llm_layer_collector import LlmLayerCollector

from language_pipes.llm_layer_collector.load_layer import get_shard_data
from language_pipes.util.utils import size_of_tensor, tensor_hash
from language_pipes.util.enums import ModelPartType
from language_pipes.llm_layer_collector.helpers import get_config


def get_avg_layer_size(model_path: Path) -> Tuple[int, str]:
    if not os.path.exists(model_path):
        return -1, ""
    collector = LlmLayerCollector(
        model_dir=model_path,
        cache_file=os.path.join(model_path, "..", "cache.json"),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    shard_data = get_shard_data(0, 0, torch.device('cpu'), collector.model_dir, collector.layer_prefix, collector.layer_files, torch.bfloat16)
    total_size = 0
    for key in collector.layer_files:
        if (collector.layer_prefix + "0.") in key:
            total_size += size_of_tensor(shard_data[key])

    lyrs = collector.load_layer_set(0, 0)

    hsh = ""
    if collector.config.model_type == "phi3":
        hsh = tensor_hash(lyrs[0].cls.self_attn.o_proj.weight)  # type: ignore
    else:
        hsh = tensor_hash(lyrs[0].cls.self_attn.q_proj.weight)  # type: ignore

    return total_size, hsh


def data_of_type(typ: ModelPartType, model_path: Path) -> Tuple[float, str]:
    config = get_config(model_path)

    size = 0
    hash = ""
    if typ == ModelPartType.EMBED:
        e = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(
            dtype=torch.bfloat16
        )
        size = size_of_tensor(e.weight)
        hash = tensor_hash(e.weight)

    if typ == ModelPartType.NORM:
        n = AutoRMSNorm(config).to(dtype=torch.bfloat16)
        size = size_of_tensor(n.cls.weight) # pyright: ignore[reportArgumentType]
        hash = tensor_hash(n.cls.weight) # pyright: ignore[reportArgumentType]
    if typ == ModelPartType.HEAD:
        h = torch.nn.Linear(config.hidden_size, config.vocab_size).to(
            dtype=torch.bfloat16
        )
        size = size_of_tensor(h.weight)
        hash = tensor_hash(h.weight)

    return size, hash


def get_computed_data(model_path: Path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found")
    computed_path = os.path.join(model_path, "meta_data.json")
    if os.path.exists(computed_path):
        with open(computed_path) as f:
            return json.load(f)

    meta_data = {}
    model_path = model_path / "data"
    size, hash = data_of_type(ModelPartType.EMBED, model_path)
    meta_data["embed_size"] = size
    meta_data["embed_hash"] = hash
    size, hash = data_of_type(ModelPartType.NORM, model_path)
    size, hash = data_of_type(ModelPartType.HEAD, model_path)
    meta_data["head_size"] = size
    meta_data["head_hash"] = hash
    size, hash = get_avg_layer_size(model_path)
    meta_data["avg_layer_size"] = size
    meta_data["layer_hashes"] = hash

    with open(computed_path, "w") as f:
        json.dump(meta_data, f)

    return meta_data


class LlmMetadata:
    embed_size: int
    head_size: int
    avg_layer_size: int

    embed_hash: str
    head_hash: str
    layer_hashes: List[str]

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            return
        data = get_computed_data(model_dir)
        self.embed_size = data["embed_size"]
        self.head_size = data["head_size"]
        self.avg_layer_size = data["avg_layer_size"]
        self.embed_hash = data["embed_hash"]
        self.head_hash = data["head_hash"]
        self.layer_hashes = data["layer_hashes"]

    def to_json(self):
        return {
            "embed_size": self.embed_size,
            "head_size": self.head_size,
            "avg_layer_size": self.avg_layer_size,
            "embed_hash": self.embed_hash,
            "head_hash": self.head_hash,
            "layer_hashes": self.layer_hashes,
        }

    @staticmethod
    def from_dict(data: dict) -> "LlmMetadata":
        c = LlmMetadata(None)
        c.embed_size = data["embed_size"]
        c.head_size = data["head_size"]
        c.avg_layer_size = data["avg_layer_size"]
        c.embed_hash = data["embed_hash"]
        c.head_hash = data["head_hash"]
        c.layer_hashes = data["layer_hashes"]
        return c


def validate_model(c1: LlmMetadata, c2: LlmMetadata):
    return (
        c1.embed_hash == c2.embed_hash
        and c1.head_hash == c2.head_hash
        and c1.layer_hashes == c2.layer_hashes
    )
