from dataclasses import dataclass
from language_pipes.modeling.llm_meta_data import LlmMetadata

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
    meta_data: LlmMetadata

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
            "meta_data": self.meta_data.to_json()
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
            meta_data=LlmMetadata.from_dict(data["meta_data"])
        )