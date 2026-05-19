from dataclasses import dataclass

from torch import Tensor
from typing import Dict, Tuple

@dataclass
class LLmComputationState:
    state: Tensor
    position_ids: Tensor
    cache_position: Tensor
    causal_mask: Dict[str, Tensor]
    position_embeddings: Dict[str, Tuple[Tensor, Tensor]]