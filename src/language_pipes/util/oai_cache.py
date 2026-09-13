from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheOptions:
    prompt_cache_key: str = ""
    mode: str = "implicit"
    ttl_seconds: int | None = None
    breakpoints: list[int] = field(default_factory=list)

class CacheOptionError(ValueError):
    pass

# Write points come from the client in explicit mode and from the end of the
# prompt in implicit mode.
MODES = ("implicit", "explicit")

# Retention values, in seconds. `None` means "no request": the node's own
# `max_cache_time` applies. `in_memory` is OpenAI's spelling for exactly that,
# and the hyphenated form is accepted because both appear in the wild.
TTL_VALUES: dict[str, int | None] = {
    "in_memory": None,
    "in-memory": None,
    "30m": 1800,
    "24h": 86400,
}

# Extras are ignored rather than rejected, which is what OpenAI does.
MAX_BREAKPOINTS = 4

def _parse_ttl(raw: Any, parameter: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in TTL_VALUES:
        allowed = "', '".join(k for k in TTL_VALUES if k != "in-memory")
        raise CacheOptionError(f"unknown {parameter} '{raw}', expected one of '{allowed}'")
    return TTL_VALUES[raw]

def _parse_mode(options: dict[str, Any]) -> str:
    raw = options.get("mode")
    if raw is None:
        return "implicit"
    if raw not in MODES:
        allowed = "', '".join(MODES)
        raise CacheOptionError(
            f"unknown prompt_cache_options.mode '{raw}', expected one of '{allowed}'"
        )
    return raw

def _breakpoint_on(block: Any) -> bool:
    """Whether one content block carries a `prompt_cache_breakpoint`."""
    if not isinstance(block, dict) or "prompt_cache_breakpoint" not in block:
        return False
    marker = block["prompt_cache_breakpoint"]
    if not isinstance(marker, dict):
        raise CacheOptionError("prompt_cache_breakpoint must be an object")
    mode = marker.get("mode", "explicit")
    if mode != "explicit":
        raise CacheOptionError(
            f"unknown prompt_cache_breakpoint mode '{mode}', expected 'explicit'"
        )
    return True

def _reject_instruction_breakpoints(instructions: Any):
    """`instructions` is a string, so it has nowhere to hang a breakpoint"""
    if instructions is None or isinstance(instructions, str):
        return
    blocks = instructions if isinstance(instructions, list) else [instructions]
    if any(_breakpoint_on(block) for block in blocks):
        raise CacheOptionError(
            "prompt_cache_breakpoint is not allowed on instructions; "
            "move the text into an input developer message and mark it there"
        )

def _marks_a_breakpoint(item: Any) -> bool:
    """Whether one `input` item ends a stable block"""
    if not isinstance(item, dict):
        return False
    # `content` on a message, `output` on a tool result
    blocks = []
    for value in (item.get("content"), item.get("output")):
        blocks.extend(value if isinstance(value, list) else [value])
    return any(_breakpoint_on(block) for block in blocks)

def _input_breakpoints(response_input: Any) -> list[int]:
    """Indices of the `input` items marked with a breakpoint, at most four."""
    items = response_input if isinstance(response_input, list) else [response_input]
    marked = [i for i, item in enumerate(items) if _marks_a_breakpoint(item)]
    return marked[:MAX_BREAKPOINTS]

def parse_cache_options(
    data: dict[str, Any],
    responses: bool = False
) -> CacheOptions:
    """Read the caching parameters off a request body"""
    raw_key = data.get("prompt_cache_key")
    prompt_cache_key = str(raw_key) if isinstance(raw_key, str) else ""
    options = CacheOptions(prompt_cache_key=prompt_cache_key)
    if not responses:
        return options

    raw_options = data.get("prompt_cache_options")
    if raw_options is not None:
        if not isinstance(raw_options, dict):
            raise CacheOptionError("prompt_cache_options must be an object")
        options.mode = _parse_mode(raw_options)
        options.ttl_seconds = _parse_ttl(raw_options.get("ttl"), "prompt_cache_options.ttl")

    # The older spelling of the same request
    retention = _parse_ttl(data.get("prompt_cache_retention"), "prompt_cache_retention")
    if options.ttl_seconds is None and retention is not None:
        options.ttl_seconds = retention

    _reject_instruction_breakpoints(data.get("instructions"))
    options.breakpoints = _input_breakpoints(data.get("input"))
    return options