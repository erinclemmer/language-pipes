from dataclasses import dataclass
from language_pipes.modeling.meta_computed import MetaComputed

@dataclass
class MetaModel:
    process_id: str
    start_layer: int
    end_layer: int
    loaded: bool
    
    node_id: str
    pipe_id: str
    model_id: str
    num_layers: int
    computed: MetaComputed

    def to_json(self):
        return {
            "process_id": self.process_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "node_id": self.node_id,
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
            node_id=data["node_id"],
            pipe_id=data["pipe_id"],
            model_id=data["model_id"],
            num_layers=data["num_layers"],
            computed=MetaComputed.from_dict(data["computed"])
        )