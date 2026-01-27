from typing import Callable, List, Optional
from pathlib import Path
from uuid import uuid4

from transformers import AutoTokenizer
from transformers.models.auto import AutoConfig
from language_pipes.network_protocol import StateNetworkNode

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.modeling.llm_model import LlmModel
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.chat import ChatMessage

class Pipe:
    pipe_id: str
    model_id: str
    segments: List[LlmModel]

    router: StateNetworkNode
    tokenizer: Callable
    
    def __init__(
            self, 
            router: StateNetworkNode,
            pipe_id: Optional[str],
            model_id: str,
            model_dir: str
        ):
        self.router = router
        self.model_id = model_id
        model_path = str(Path(model_dir) / model_id / 'data')
        
        if pipe_id is None:
            self.pipe_id = str(uuid4())
        else:
            self.pipe_id = pipe_id

        self.segments = []
        self.tokenizer = lambda: AutoTokenizer.from_pretrained(model_path)

    def send_job(self, job: NetworkJob, node_id: str):
        data = job.to_bytes()
        if node_id == self.router.node_id():
            self.router._receive_data(data)
        else:
            self.router.send_to_node(node_id, data)

    def tokenize(self, prompt: Optional[str], messages: List[ChatMessage]) -> List[int]:
        tokenizer: AutoTokenizer = self.tokenizer()
        if prompt is None:
            prompt = tokenizer.apply_chat_template([m.to_json() for m in messages], tokenize=False, add_generation_prompt=True, chat_template=tokenizer.chat_template)
        return [int(t) for t in tokenizer.encode(prompt, return_tensors='pt')[0].numpy()]

    def get_layer(self, layer: int, need_physical: bool = False) -> Optional[LlmModel]:
        for segment in self.segments:
            if segment.start_layer == layer and (not need_physical or not segment.virtual):
                return segment
        return None
    
    def num_hidden_layers(self) -> Optional[int]:
        for segment in self.segments:
            if not segment.virtual:
                return segment.num_hidden_layers
        return None

    def get_computed(self):
        return self.segments[0].meta_data

    def sort_segments(self):
        self.segments = sorted(self.segments, key=lambda x: x.start_layer)

    def is_complete(self):
        if len(self.segments) == 0:
            return False
        
        self.sort_segments()
        current_layer = 0
        for s in self.segments:
            if not s.loaded:
                break
            if s.start_layer == current_layer:
                current_layer = s.end_layer + 1

        return current_layer == self.num_hidden_layers()

    @staticmethod
    def from_meta(
        meta_pipe: MetaPipe, 
        hosted_models: List[LlmModel], 
        router: StateNetworkNode,
        model_dir: str
    ) -> 'Pipe':
        p = Pipe(
            model_id=meta_pipe.model_id, 
            pipe_id=meta_pipe.pipe_id,
            model_dir=model_dir,
            router=router
        )
        local_segments = []
        for model in hosted_models:
            if model.pipe_id == meta_pipe.pipe_id:
                p.segments.append(model)
                local_segments.append(model.process_id)
        p.segments.extend([LlmModel.from_meta(s, model_dir) for s in meta_pipe.segments if s.process_id not in local_segments])
        p.sort_segments()
        return p
