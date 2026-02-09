import torch
from typing import Optional, Dict, Tuple

class LLmComputationState:
    state: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    position_embeddings_local: torch.Tensor
    position_embeddings_global: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    causal_mask: Dict[str, torch.Tensor]

    def __init__(self):
        self.state = None # type: ignore
        self.position_embeddings = None # type: ignore
        self.position_embeddings_local = None # type: ignore
        self.position_embeddings_global = None # type: ignore
        self.position_ids = None # type: ignore
        self.cache_position = None # type: ignore
        self.causal_mask = { }