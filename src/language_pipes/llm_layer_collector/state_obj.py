from dataclasses import dataclass, field

from torch import Tensor
from typing import Dict, Tuple, Optional


@dataclass
class LLmComputationState:
    state: Tensor
    position_ids: Tensor
    cache_position: Tensor
    causal_mask: Dict[str, Optional[Tensor]] # Needs to be optional because sometimes the mask is None
    position_embeddings: Dict[str, Tuple[Tensor, Tensor]]
    # Gemma4 Per-Layer Embeddings (PLE): pre-computed read-only tensor of shape
    # [batch, seq, num_hidden_layers, hidden_size_per_layer_input]. Computed once on
    # the embedding node and sliced per-layer. None for models without PLE.
    per_layer_inputs: Optional[Tensor] = None
    # Gemma4 cross-node KV sharing: transient per-forward dict keyed by layer_type. The
    # last non-shared layer of each type writes full-length (k, v); the shared layers
    # read it. Mutated as it flows forward, so it is written back through
    # compute_layers / Job.set_layer to propagate to the next node. Empty for other models.
    shared_kv_states: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)