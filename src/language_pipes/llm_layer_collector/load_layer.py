import gc
import re
from pathlib import Path
import warnings
import torch
from typing import List, Dict, Set

from safetensors import safe_open
from transformers.configuration_utils import PretrainedConfig
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.helpers import get_shard_keys, get_shard_tensor

# Matches per-expert checkpoint weights, e.g. "mlp.experts.7.gate_proj.weight".
EXPERT_WEIGHT_RE = re.compile(r"^(.*experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

def check_8bit_support() -> None:
    """Fail fast with an actionable message before any shards are read."""
    try:
        with warnings.catch_warnings():
            # bitsandbytes imports trigger a torch.jit.script_method
            # DeprecationWarning on Python 3.14+; it's not actionable here.
            warnings.simplefilter("ignore", DeprecationWarning)
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError as e:
        raise ImportError(
            "LP_8_BIT_MODE requires the bitsandbytes package. "
            "Install it with: pip install bitsandbytes"
        ) from e


def replace_linear_with_8bit(module: torch.nn.Module) -> None:
    """Swap every ``nn.Linear`` in the module tree for a bitsandbytes
    ``Linear8bitLt`` (LLM.int8).

    Weights must be on CPU when this runs — bitsandbytes quantizes them during
    the subsequent move to a CUDA device. Fused MoE expert tensors are plain
    parameters rather than ``nn.Linear`` modules, so they stay in fp16.
    """
    import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            quantized = bnb.nn.modules.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            quantized.weight = bnb.nn.modules.Int8Params(
                child.weight.data.to(torch.float16),
                requires_grad=False
            )
            if child.bias is not None:
                quantized.bias = torch.nn.Parameter(
                    child.bias.data.to(torch.float16), requires_grad=False
                )
            setattr(module, name, quantized)
        else:
            replace_linear_with_8bit(child)


def fuse_moe_expert_weights(
        layer_state_dict: Dict[str, torch.Tensor],
        model_keys: Set[str],
    ) -> None:
    """Fuse per-expert checkpoint weights into the stacked 3D tensors that recent
    transformers MoE modules expect.

    HuggingFace MoE checkpoints store each expert separately as
    ``mlp.experts.{i}.gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``,
    but the in-memory module fuses them into ``mlp.experts.gate_up_proj`` and
    ``mlp.experts.down_proj``. ``from_pretrained`` performs this conversion; because we
    load layer state dicts directly with ``load_state_dict(strict=False)``, the
    per-expert keys would otherwise be silently dropped and the experts would run on
    uninitialized memory (producing garbage output with no error).

    Mutates ``layer_state_dict`` in place, replacing the per-expert keys with the fused
    parameters. Only runs when the module actually expects the fused parameter, so
    non-MoE layers and modules that keep per-expert weights are left untouched.
    """
    # prefix -> proj_name -> {expert_idx: tensor}
    grouped: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    for key in list(layer_state_dict.keys()):
        match = EXPERT_WEIGHT_RE.match(key)
        if match is None:
            continue
        prefix, idx, proj = match.group(1), int(match.group(2)), match.group(3)
        if f'{prefix}.gate_up_proj' not in model_keys:
            continue
        grouped.setdefault(prefix, {}).setdefault(proj, {})[idx] = layer_state_dict.pop(key)

    for prefix, projs in grouped.items():
        def stacked(proj_name: str) -> torch.Tensor:
            by_idx = projs[proj_name]
            return torch.stack([by_idx[i] for i in sorted(by_idx)], dim=0)

        # gate_up_proj[e] = [gate_proj; up_proj] along the output dim, matching the
        # module's `linear(...).chunk(2, dim=-1)` split into (gate, up).
        layer_state_dict[f'{prefix}.gate_up_proj'] = torch.cat(
            [stacked('gate_proj'), stacked('up_proj')], dim=1
        )
        layer_state_dict[f'{prefix}.down_proj'] = stacked('down_proj')

def dequantize_fp8_weights(shard_data: Dict[str, torch.Tensor]) -> None:
    """Apply fp8 checkpoint scales to their weights, in place.

    FP8-quantized checkpoints (config.quantization_config.quant_method == "fp8")
    store each projection as a float8_e4m3fn ``<name>.weight`` plus a
    ``<name>.weight_scale_inv`` dequant multiplier — a scalar for per-tensor
    quantization or a 2D grid for block-wise. The weights were already cast to
    the target dtype at read time (exact for e4m3), so multiplying by the scale
    recovers the real values. ``activation_scale`` tensors are only meaningful
    to fp8 runtime kernels and are dropped so they don't reach load_state_dict.
    """
    for key in [k for k in shard_data if k.endswith('.weight_scale_inv')]:
        scale = shard_data.pop(key)
        weight_key = key[:-len('_scale_inv')]
        weight = shard_data[weight_key]
        if scale.dim() >= 2:
            scale = scale.repeat_interleave(weight.shape[0] // scale.shape[0], 0)
            scale = scale.repeat_interleave(weight.shape[1] // scale.shape[1], 1)
        shard_data[weight_key] = weight * scale
    for key in [k for k in shard_data if k.endswith('.activation_scale')]:
        del shard_data[key]

def files_to_load_for_layer(
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
    ) -> List[str]:
    files_to_load: List[str] = []
    for key in layer_file_cache.keys():
        if key.startswith(layer_prefix) and layer_file_cache[key] not in files_to_load:
            files_to_load.append(layer_file_cache[key])
    if len(files_to_load) == 0:
        raise Exception("Could not find layer data for layer prefix " + layer_prefix)
    return files_to_load

def files_to_load_for_layers(
        start_layer: int,
        end_layer: int,
        layer_prefix: str,
        layer_file_cache: Dict[str, str]
    ) -> List[str]:
    files_to_load: List[str] = []
    for i in range(start_layer, end_layer+1):
        for f in files_to_load_for_layer(f'{layer_prefix}{i}.', layer_file_cache):
            if f not in files_to_load:
                files_to_load.append(f)
    return files_to_load

def get_shard_data(
        start_layer: int,
        end_layer: int,
        device: torch.device,
        model_dir: Path,
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
    prefixes = [f'{layer_prefix}{i}.' for i in range(start_layer, end_layer+1)]
    shard_data: Dict[str, torch.Tensor] = { }
    for file_path in files_to_load_for_layers(start_layer, end_layer, layer_prefix, layer_file_cache):
        full_path = f'{model_dir}/{file_path}'
        shard = safe_open(full_path, framework='pt', device=str(device))
        for key in get_shard_keys(shard):
            for prefix in prefixes:
                if key.startswith(prefix):
                    shard_data[key] = get_shard_tensor(shard, key).detach().to(dtype)
        del shard
        gc.collect()

    dequantize_fp8_weights(shard_data)

    return shard_data

def load_layer(
        config: PretrainedConfig,
        idx: int,
        shard_data: Dict[str, torch.Tensor],
        layer_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
        load_in_8bit: bool = False
    ) -> AutoDecoderLayer:
    # In 8-bit mode weights are assembled on CPU so that bitsandbytes can
    # quantize them during the final move to the CUDA device.
    build_device = torch.device('cpu') if load_in_8bit else device
    torch.set_default_device('meta')
    lyr = AutoDecoderLayer(config, idx)
    torch.set_default_device(build_device)
    lyr = lyr.to_empty(device=build_device)

    layer_key_prefix = f'{layer_prefix}{idx}.'
    layer_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in shard_data.items():
        if key.startswith(layer_key_prefix):
            param_name = key[len(layer_key_prefix):]
            layer_state_dict[param_name] = value

    if len(layer_state_dict) == 0:
        raise Exception(f"Could not find data for layer {idx}")

    fuse_moe_expert_weights(layer_state_dict, set(lyr.cls.state_dict().keys())) # pyright: ignore[reportUnknownArgumentType]

    lyr.cls.load_state_dict(layer_state_dict, strict=False) # pyright: ignore[reportUnknownMemberType]

    if load_in_8bit:
        replace_linear_with_8bit(lyr.cls)

    return lyr.to(device, dtype)

def load_layers(
        start_layer: int,
        end_layer: int,
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        config: PretrainedConfig,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        load_in_8bit: bool = False
    ) -> List[AutoDecoderLayer]:
    if load_in_8bit:
        check_8bit_support()
    load_device = torch.device('cpu') if load_in_8bit else device
    torch.set_default_device(load_device)
    shard_data = get_shard_data(start_layer, end_layer, load_device, model_dir, layer_prefix, layer_file_cache, dtype)
    layers: List[AutoDecoderLayer] = []
    for i in range(start_layer, end_layer+1):
        layers.append(load_layer(config, i, shard_data, layer_prefix, device, dtype, load_in_8bit))

    torch.set_default_device('cpu')
    return layers
