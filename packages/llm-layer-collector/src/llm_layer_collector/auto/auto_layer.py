import importlib
import torch
from typing import Optional

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

mapper = { # type: ignore
    "llama": LlamaDecoderLayer,
    "phi3": Phi3DecoderLayer,
    "qwen3": Qwen3DecoderLayer,
    "gemma3_text": Gemma3DecoderLayer,
    "gemma4_text": Gemma4TextDecoderLayer,
    "gemma4_unified_text": Gemma4UnifiedTextDecoderLayer,
    "qwen3_moe": Qwen3MoeDecoderLayer,
    "ministral3": Ministral3DecoderLayer,
    "gpt_oss": GptOssDecoderLayer,
}

def getClass(config: PretrainedConfig) -> GradientCheckpointingLayer:
    return mapper[config.model_type] # type: ignore

def supports_sdpa(config: PretrainedConfig) -> bool:
    """Whether this architecture's attention is correct under the sdpa kernel.

    Some architectures pass extra tensors to the attention interface that only the
    eager path consumes — gpt_oss forwards its attention sinks as ``s_aux``, which
    the generic ``sdpa_attention_forward`` accepts into ``**kwargs`` and ignores.
    The forward then runs without error but produces wrong values, so the choice
    has to come from the architecture rather than a global default.

    Read off the ``*PreTrainedModel`` class that lives alongside the decoder layer
    (its module is already imported by the mapper above), so new registrations pick
    this up without another table to maintain.
    """
    module = importlib.import_module(getClass(config).__module__) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    for value in vars(module).values():
        if (isinstance(value, type) and issubclass(value, PreTrainedModel)
                and value is not PreTrainedModel):
            return bool(getattr(value, '_supports_sdpa', True))
    return True

class AutoDecoderLayer:
    cls: GradientCheckpointingLayer

    def __init__(self, config: PretrainedConfig, layer_index: int):
        self.config = config
        if self.config._attn_implementation_internal is None: # type: ignore
            self.config._attn_implementation = "eager" # type: ignore
        self.cls = getClass(self.config)(self.config, layer_index)

    def to_empty(self, device: Optional[torch.device]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: torch.device, dtype: torch.dtype) -> 'AutoDecoderLayer':
        self.cls = self.cls.to(device=device, dtype=dtype)
        return self