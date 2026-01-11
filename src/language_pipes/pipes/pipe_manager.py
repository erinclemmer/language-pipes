from typing import Optional

from language_pipes.config import LpConfig

from language_pipes.pipes.pipe import Pipe
from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import RouterPipes

from language_pipes.modeling.model_manager import ModelManager

class PipeManager:
    config: LpConfig
    router_pipes: RouterPipes
    model_manager: ModelManager

    def __init__(
        self,
        config: LpConfig,
        model_manager: ModelManager,
        router_pipes: RouterPipes
    ):
        self.config = config
        self.model_manager = model_manager
        self.router_pipes = router_pipes

    def _get_pipe_from_meta(self, meta_pipe: MetaPipe) -> Pipe:
        return Pipe.from_meta(
            meta_pipe=meta_pipe,
            hosted_models=self.model_manager.models,
            router=self.router_pipes.router,
            model_dir=self.config.model_dir
        )

    def get_pipe_by_pipe_id(self, pipe_id: str) -> Optional[Pipe]:
        meta_pipe = self.router_pipes.get_pipe_by_pipe_id(pipe_id)
        if meta_pipe is None:
            return None
        return self._get_pipe_from_meta(meta_pipe)

    def get_pipe_by_model_id(self, model_id: str) -> Optional[Pipe]:
        meta_pipe = self.router_pipes.get_pipe_by_model_id(model_id)
        if meta_pipe is None:
            return None
        return self._get_pipe_from_meta(meta_pipe)
