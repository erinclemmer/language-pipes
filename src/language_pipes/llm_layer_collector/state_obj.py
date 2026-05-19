from dataclasses import dataclass

from torch import Tensor
from typing import Dict, Tuple, Optional


@dataclass
class LLmComputationState:
    state: Tensor
    position_ids: Tensor
    cache_position: Tensor
    causal_mask: Dict[str, Optional[Tensor]] # Needs to be optional because sometimes the mask is None
    position_embeddings: Dict[str, Tuple[Tensor, Tensor]]