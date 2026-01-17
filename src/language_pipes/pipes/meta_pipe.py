from typing import List
from dataclasses import dataclass
from language_pipes.modeling.meta_model import MetaModel
from language_pipes.modeling.llm_meta_data import LlmMetadata

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

    def get_computed(self) -> LlmMetadata:
        return self.segments[0].meta_data

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
                filled_slots[i] = 2 if segment.loaded else 1
        return filled_slots

    def print_pipe(self):
        filled_slots = self.get_filled_slots()
        num_layers = len(filled_slots)
        pipe_pieces = ["|>"]
        for slot in filled_slots:
            match (slot):
                case 0:
                    pipe_pieces.append("X")
                case 1:
                    pipe_pieces.append("|")
                case 2:
                    pipe_pieces.append("=")
        pipe_pieces.append("<|")
        return "".join(pipe_pieces)

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
            if filled_slots[end_layer] > 0:
                return end_layer - 1
        return -1

    def peers(self) -> List[str]:
        peers: List[str] = []
        for segment in self.segments:
            if segment.node_id not in peers:
                peers.append(segment.node_id)
        return peers

    def is_complete(self):
        self.sort_segments()
        current_layer = 0
        for s in self.segments:
            if s.start_layer == -1 or not s.loaded:
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
Segments: {', '.join([s.node_id for s in self.segments])}
{self.print_pipe()}
End Layer: {self.segments[-1].end_layer} / {self.num_layers() - 1}
Complete: {self.is_complete()}
=================================
''')
