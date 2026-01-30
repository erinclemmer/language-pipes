from uuid import uuid4
from logging import Logger
from typing import List, Optional, Tuple, Dict

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import RouterPipes

from language_pipes.modeling.llm_model import LlmModel
from language_pipes.modeling.end_model import EndModel
from language_pipes.modeling.llm_meta_data import validate_model

from language_pipes.config import LpConfig

class ModelManager:
    logger: Logger
    config: LpConfig
    router_pipes: RouterPipes
    models: List[LlmModel]
    end_models: List[EndModel]
    pipes_hosted: Dict[str, List[str]]

    def __init__(
        self, 
        logger: Logger,
        config: LpConfig,
        router_pipes: RouterPipes
    ):
        self.config = config
        self.logger = logger
        self.router_pipes = router_pipes
        self.models = []
        self.end_models = []
        self.pipes_hosted = { }
        for m in self.config.hosted_models:
            self._host_model(m.id, m.max_memory, m.device, m.load_ends)

    def stop(self):
        for m in self.models:
            m.cleanup_tensors()
        for m in self.end_models:
            m.clean_up()
        self.models = []
        self.end_models = []

    def get_end_model(self, model_id: str) -> Optional[EndModel]:
        for m in self.end_models:
            if m.model_id == model_id:
                return m
        return None

    def _get_model_for_pipe(self, model_id: str, pipe: MetaPipe, device: str, available_memory: int | float) -> Tuple[int | float, Optional[LlmModel]]:
        start_memory = available_memory

        new_model: Optional[LlmModel] = LlmModel.from_id(
            model_dir=self.config.model_dir,
            model_id=model_id,
            node_id=self.config.node_id,
            pipe_id=pipe.pipe_id,
            device=device
        )
        if new_model is None:
            return None
        meta_data = new_model.meta_data
        if self.config.model_validation and len(pipe.segments) > 0 and not validate_model(new_model.meta_data, pipe.get_computed()):
            self.logger.warning(f'Computed data for model {model_id} does not match')
            return available_memory, None
        
        num_layers_to_load = int(available_memory // meta_data.avg_layer_size) - 1
        total_layers = new_model.collector.config.num_hidden_layers
        start_layer = pipe.next_start_layer()
        if num_layers_to_load == -1:
            start_layer = -1
            end_layer = -1
        else:
            end_layer = min([start_layer + num_layers_to_load, pipe.next_end_layer(total_layers), new_model.num_hidden_layers]) if start_layer != -1 else -1
            available_memory = available_memory - (end_layer - start_layer + 1) * meta_data.avg_layer_size

        if num_layers_to_load > -1 and end_layer != -1 and start_layer != -1:
            self.logger.info(f'Using {(start_memory - available_memory) / 10**9:.2f} GB of memory to load model {model_id}')
            new_model.start_layer = start_layer
            new_model.end_layer = end_layer
            new_model.print(self.logger)
        else:
            new_model = None
        return available_memory, new_model

    def _load_end_model(self, model_id: str, device: str):
        model = EndModel(self.config.model_dir, model_id, device)
        self.end_models.append(model)
        return model

    def _host_model(self, model_id: str, max_memory: float, device: str, load_ends: bool):
        available_memory = max_memory * 10 ** 9
        models_to_load: List[LlmModel] = []
        end_model = None
        if load_ends:
            end_model = self._load_end_model(model_id, device)
        
        if model_id not in self.pipes_hosted:
            self.pipes_hosted[model_id] = []
        
        for pipe_id in [p.pipe_id for p in self.router_pipes.pipes_for_model(model_id, False)]:
            if pipe_id not in self.pipes_hosted[model_id] and len(self.pipes_hosted[model_id]) >= self.config.max_pipes:
                break
            loaded = True
            while loaded:
                pipe = self.router_pipes.get_pipe_by_pipe_id(pipe_id)
                if pipe is None: 
                    break
                available_memory, model = self._get_model_for_pipe(model_id, pipe, device, available_memory)
                loaded = model is not None
                if model is not None:
                    self.pipes_hosted[model_id].append(model.pipe_id)
                    self.router_pipes.add_model_to_network(model.to_meta())
                    models_to_load.append(model)

        if len(self.pipes_hosted[model_id]) < self.config.max_pipes:
            new_pipe = MetaPipe(str(uuid4()), model_id, [])
            self.pipes_hosted[model_id].append(new_pipe.pipe_id)
            _, model = self._get_model_for_pipe(model_id, new_pipe, device, available_memory)
            if model is not None:
                self.router_pipes.add_model_to_network(model.to_meta())
                models_to_load.append(model)

        if load_ends and end_model is not None:
            end_model.load()

        for m in models_to_load:
            m.load()
            self.router_pipes.update_model(m.to_meta())
            self.models.append(m)
