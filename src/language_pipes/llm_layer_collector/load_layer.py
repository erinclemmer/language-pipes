import gc
import re
from pathlib import Path
import torch
from typing import List, Dict, Set

from safetensors import safe_open
from transformers.configuration_utils import PretrainedConfig
from language_pipes.llm_layer_collector.auto.auto_layer import AutoDecoderLayer
from language_pipes.llm_layer_collector.helpers import get_shard_keys, get_shard_tensor

# Matches per-expert checkpoint weights, e.g. "mlp.experts.7.gate_proj.weight".
EXPERT_WEIGHT_RE = re.compile(r"^(.*experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


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
    
    return shard_data

def load_layer(
        config: PretrainedConfig, 
        idx: int, 
        shard_data: Dict[str, torch.Tensor],
        layer_prefix: str,
        device: torch.device,
        dtype: torch.dtype
    ) -> AutoDecoderLayer:
    torch.set_default_device('meta')
    lyr = AutoDecoderLayer(config, idx)
    torch.set_default_device(device)
    lyr = lyr.to_empty(device=device)

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

    return lyr.to(device, dtype)

def load_layers(
        start_layer: int, 
        end_layer: int, 
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        config: PretrainedConfig,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype
    ) -> List[AutoDecoderLayer]:
    torch.set_default_device(device)
    shard_data = get_shard_data(start_layer, end_layer, device, model_dir, layer_prefix, layer_file_cache, dtype)
    layers: List[AutoDecoderLayer] = []
    for i in range(start_layer, end_layer+1):
        layers.append(load_layer(config, i, shard_data, layer_prefix, device, dtype))

    torch.set_default_device('cpu')
    return layers
