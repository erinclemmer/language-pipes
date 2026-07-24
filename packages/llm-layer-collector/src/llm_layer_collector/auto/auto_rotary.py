from typing import Optional, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextRotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

mapper = { # pyright: ignore[reportUnknownVariableType]
    "llama": LlamaRotaryEmbedding,
    "phi3": Phi3RotaryEmbedding,
    "qwen3": Qwen3RotaryEmbedding,
    "gemma3_text": Gemma3RotaryEmbedding,
    "gemma4_text": Gemma4TextRotaryEmbedding,
    "gemma4_unified_text": Gemma4UnifiedTextRotaryEmbedding,
    "qwen3_moe": Qwen3MoeRotaryEmbedding,
    "ministral3": Ministral3RotaryEmbedding,
    "gpt_oss": GptOssRotaryEmbedding
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