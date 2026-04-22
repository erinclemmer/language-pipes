from dataclasses import dataclass

from torch import Tensor
from typing import Dict, Optional, Tuple

@dataclass
class LLmComputationState:
    state: Tensor
    position_embeddings: Optional[Tuple[Tensor, Tensor]]
    position_embeddings_local: Optional[Tuple[Tensor, Tensor]]
    position_embeddings_global: Optional[Tuple[Tensor, Tensor]]
    position_ids: Tensor
    cache_position: Tensor
    causal_mask: Dict[str, Optional[Tensor]]