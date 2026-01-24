from typing import Tuple

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

mapper = {
    "llama": LlamaRotaryEmbedding,
    "qwen3": Qwen3RotaryEmbedding,
    "gemma3_text": Gemma3RotaryEmbedding,
    "qwen3_moe": Qwen3MoeRotaryEmbedding
}

def getClass(config: PretrainedConfig) -> torch.nn.Module:
    return mapper[config.model_type]

class AutoRotaryEmbedding:
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.cls = getClass(config)(config)
        self.cls.inv_freq = self.cls.inv_freq.to(torch.float16)

    def __call__(self, x, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls(x, position_ids)