import json
from typing import List, Dict, Optional
from distributed_state_network import DSNode

from language_pipes.util.meta import MetaModel, MetaPipe

def aggregate_models(models: List[MetaModel]) -> List[MetaPipe]:
    pipes: List[MetaPipe] = []
    pipe_ids: List[str] = []
    for model in models:
        if model.pipe_id in pipe_ids:
            existing_pipe = [p for p in pipes if p.pipe_id == model.pipe_id][0]
            existing_pipe.segments.append(model)
        else:
            pipes.append(MetaPipe(model.pipe_id, model.model_id, [model]))
            pipe_ids.append(model.pipe_id)

    for pipe in pipes:
        pipe.sort_segments()
    return pipes

class RouterPipes:
    router: DSNode

    def __init__(self, router: DSNode):
        self.router = router

    def add_model_to_network(self, model: MetaModel):
        models_string = self.router.read_data(self.router.config.node_id, 'models') or '[]'
        current_models = [MetaModel.from_dict(m) for m in json.loads(models_string)]
        current_models.append(model)
        self.router.update_data(
            'models',
            json.dumps([m.to_json() for m in current_models])
        )

    def update_model(self, model: MetaModel):
        models_string = self.router.read_data(self.router.config.node_id, 'models') or '[]'
        current_models = [MetaModel.from_dict(m) for m in json.loads(models_string)]
        matching_models = [m for m in current_models if m.process_id == model.process_id]
        if len(matching_models) < 1:
            raise Exception("Could not update model")
        current_models.remove(matching_models[0])
        current_models.append(model)
        self.router.update_data(
            'models',
            json.dumps([m.to_json() for m in current_models])
        )

    def all_models(self) -> List[MetaModel]:
        network_models: Dict[str, List[MetaModel]] = { }
        for peer in self.router.peers():
            peer_models = json.loads(self.router.read_data(peer, 'models') or '[]')
            network_models[peer] = []
            for m in peer_models:
                network_models[peer].append(MetaModel.from_dict(m))
        models: List[MetaModel] = []
        for key in network_models.keys():
            models.extend(network_models[key])
        return models

    def pipes_for_model(self, model_id: str, find_completed: bool) -> List[MetaPipe]:
        models: List[MetaModel] = []
        for model in self.all_models():
            if model.model_id != model_id:
                continue
            models.append(model)

        return [pipe for pipe in aggregate_models(models) if pipe.is_complete() == find_completed]

    def network_pipes(self) -> List[MetaPipe]:
        return aggregate_models(self.all_models())

    def network_pipe(self, pipe_id: str) -> Optional[MetaPipe]:
        models: List[MetaModel] = []
        for model in self.all_models():
            if model.pipe_id == pipe_id:
                models.append(model)
        if len(models) == 0:
            return None
        
        return aggregate_models(models)[0]
