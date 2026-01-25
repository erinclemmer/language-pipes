import torch
from typing import Optional, Dict

class LLmComputationState:
    state: torch.Tensor
    position_embeddings: Optional[torch.Tensor]
    position_embeddings_local: Optional[torch.Tensor]
    position_embeddings_global: Optional[torch.Tensor]
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    causal_mask: Dict

    def __init__(self):
        self.state = None
        self.position_embeddings = None
        self.position_embeddings_local = None
        self.position_embeddings_global = None
        self.position_ids = None
        self.cache_position = None
        self.causal_mask = { }