from typing import List
from dataclasses import dataclass

from transformers.models.auto import AutoConfig

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

@dataclass
class MetaModel:
    process_id: str
    start_layer: int
    end_layer: int
    loaded: bool
    
    router_id: str
    pipe_id: str
    model_id: str
    num_layers: int
    computed: MetaComputed

    def to_json(self):
        return {
            "process_id": self.process_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "router_id": self.router_id,
            "pipe_id": self.pipe_id,
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "loaded": self.loaded,
            "computed": self.computed.to_json()
        }

    @staticmethod
    def from_dict(data: dict):
        return MetaModel(
            process_id=data["process_id"],
            start_layer=data["start_layer"],
            end_layer=data["end_layer"],
            loaded=data["loaded"],
            router_id=data["router_id"],
            pipe_id=data["pipe_id"],
            model_id=data["model_id"],
            num_layers=data["num_layers"],
            computed=MetaComputed.from_dict(data["computed"])
        )

@dataclass
class MetaPipe:
    pipe_id: str
    model_id: str

    segments: List[MetaModel]

    def num_layers(self):
        if len(self.segments) == 0:
            return 0
        else:
            return self.segments[0].num_layers

    def is_loading(self) -> bool:
        return len([s for s in self.segments if not s.loaded]) > 0

    def get_computed(self) -> MetaComputed:
        return self.segments[0].computed

    def sort_segments(self):
        self.segments = sorted(self.segments, key=lambda x: x.start_layer)

    def get_filled_slots(self, num_layers: int | None = None):
        if num_layers is None:
            num_layers = self.num_layers()
        filled_slots = [0 for _ in range(0, num_layers)]
        for segment in self.segments:
            if segment.start_layer == -1:
                continue
            for i in range(segment.start_layer, min([segment.end_layer + 1, num_layers])):
                filled_slots[i] = 1
        return filled_slots

    def next_start_layer(self) -> int:
        if len(self.segments) == 0:
            return 0
        filled_slots = self.get_filled_slots()
        for slot in range(0, self.num_layers()):
            if filled_slots[slot] == 0:
                return slot
        return -1

    def next_end_layer(self, num_layers: int | None = None) -> int:
        if num_layers is None:
            num_layers = self.num_layers()
        start = self.next_start_layer()
        filled_slots = self.get_filled_slots(num_layers)
        for end_layer in range(start, num_layers):
            if end_layer == num_layers - 1:
                return end_layer
            if filled_slots[end_layer] == 1:
                return end_layer - 1
        return -1

    def peers(self) -> List[str]:
        peers: List[str] = []
        for segment in self.segments:
            if segment.router_id not in peers:
                peers.append(segment.router_id)
        return peers

    def is_complete(self):
        self.sort_segments()
        current_layer = 0
        for s in self.segments:
            if s.start_layer == -1:
                continue
            if s.start_layer == current_layer:
                current_layer = s.end_layer + 1

        return current_layer == self.segments[0].num_layers

    def print(self, logger):
        self.sort_segments()
        logger.info(f'''
=================================
Pipe Status:
Model ID: {self.model_id}
Pipe: {self.pipe_id}
Segments: {', '.join([s.router_id for s in self.segments])}
{self.get_filled_slots()}
End Layer: {self.segments[-1].end_layer} / {self.num_layers() - 1}
Complete: {self.is_complete()}
=================================
''')
