from typing import Callable, Optional, List

from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.pipes.meta_pipe import MetaPipe

class PipeProvider:
    get_pipe_manager: Callable[[], Optional[PipeManager]]

    def __init__(self, get_pipe_manager: Callable):
        self.get_pipe_manager = get_pipe_manager

    def get_connected_pipes(self) -> Optional[List[MetaPipe]]:
        pipe_manager = self.get_pipe_manager()
        if pipe_manager is None:
            return None 
        if len(pipe_manager.model_manager.pipes_hosted.keys()) == 0:
            return []
        
        pipes: List[MetaPipe] = []
        for key in pipe_manager.model_manager.pipes_hosted.keys():
            pipe_ids = pipe_manager.model_manager.pipes_hosted[key]
            for pipe_id in pipe_ids:
                pipe = pipe_manager.router_pipes.get_pipe_by_pipe_id(pipe_id)
                if pipe is not None:
                    pipes.append(pipe)
        
        return pipes
    
    def get_network_pipes(self) -> Optional[List[MetaPipe]]:
        pipe_manager = self.get_pipe_manager()
        if pipe_manager is None:
            return None
        
        return pipe_manager.router_pipes._network_pipes()