import torch
from typing import Dict, Optional, Tuple

class LLmComputationState:
    state: Optional[torch.Tensor]
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]
    position_embeddings_local: Optional[Tuple[torch.Tensor, torch.Tensor]]
    position_embeddings_global: Optional[Tuple[torch.Tensor, torch.Tensor]]
    position_ids: Optional[torch.Tensor]
    cache_position: Optional[torch.Tensor]
    causal_mask: Dict[str, Optional[torch.Tensor]]

    def __init__(self):
        self.state = None # type: ignore
        self.position_embeddings = None # type: ignore
        self.position_embeddings_local = None # type: ignore
        self.position_embeddings_global = None # type: ignore
        self.position_ids = None # type: ignore
        self.cache_position = None # type: ignore
        self.causal_mask = { }