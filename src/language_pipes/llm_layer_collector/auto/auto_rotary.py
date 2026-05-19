from typing import Optional, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
from transformers.models.glm4v.modeling_glm4v import Glm4vTextRotaryEmbedding

mapper = { # pyright: ignore[reportUnknownVariableType]
    "llama": LlamaRotaryEmbedding,
    "phi3": Phi3RotaryEmbedding,
    "qwen3": Qwen3RotaryEmbedding,
    "gemma3_text": Gemma3RotaryEmbedding,
    "qwen3_moe": Qwen3MoeRotaryEmbedding,
    "glm4v": Glm4vTextRotaryEmbedding
}

def getClass(config: PretrainedConfig) -> torch.nn.Module:
    return mapper[config.model_type] # pyright: ignore[reportUnknownVariableType]

class AutoRotaryEmbedding:
    cls: torch.nn.Module

    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.cls = getClass(config)(config)

    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_type is not None:
            return self.cls(x, position_ids, layer_type)
        return self.cls(x, position_ids)