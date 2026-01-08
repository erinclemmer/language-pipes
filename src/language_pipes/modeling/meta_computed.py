from typing import List
from dataclasses import dataclass

@dataclass
class MetaComputed:
    embed_size: int
    head_size: int
    avg_layer_size: int

    embed_hash: str
    head_hash: str
    layer_hashes: List[str]

    def to_json(self):
        return {
            "embed_size": self.embed_size,
            "head_size": self.head_size,
            "avg_layer_size": self.avg_layer_size,
            "embed_hash": self.embed_hash,
            "head_hash": self.head_hash,
            "layer_hashes": self.layer_hashes
        }

    @staticmethod
    def from_dict(data: dict):
        return MetaComputed(
            embed_size=data["embed_size"],
            head_size=data["head_size"],
            avg_layer_size=data["avg_layer_size"],
            embed_hash=data["embed_hash"],
            head_hash=data["head_hash"],
            layer_hashes=data["layer_hashes"]
        )