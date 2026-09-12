"""Prompt-caching request options and usage reporting for the OpenAI surface.

Kept out of `oai.py` because the parsing rule that matters here is a privacy
rule, not a serialization one: on a server with no `api_keys` configured every
caller shares one identity, so `prompt_cache_key` is the only thing separating
them and a request without one must not touch the cache at all.

The Responses endpoint takes the full surface - `prompt_cache_options`,
`prompt_cache_retention` and per-block breakpoints. Chat completions take only
`prompt_cache_key`, which is all OpenAI exposes there, and silently ignore the
rest rather than rejecting a request the server accepted before.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Write points come from the client in explicit mode and from the end of the
# prompt in implicit mode.
MODES = ("implicit", "explicit")

# Retention values, in seconds. `None` means "no request": the node's own
# `max_cache_time` applies. `in_memory` is OpenAI's spelling for exactly that,
# and the hyphenated form is accepted because both appear in the wild.
TTL_VALUES: Dict[str, Optional[int]] = {
    "in_memory": None,
    "in-memory": None,
    "30m": 1800,
    "24h": 86400,
}

# Extras are ignored rather than rejected, which is what OpenAI does.
MAX_BREAKPOINTS = 4


class CacheOptionError(ValueError):
    """A prompt-caching parameter the server cannot honor.

    Raised at parse time and turned into a `400` by `oai_responses_create`, the
    same way an unknown enum value is rejected by OpenAI rather than silently
    ignored - a client that asked for a 24h TTL and got the node default would
    otherwise have no way to find out.
    """


@dataclass
class CacheOptions:
    # Partitions the cache within an API key. On an unauthenticated server it is
    # also the only thing partitioning at all, which is why an absent value
    # disables caching there.
    prompt_cache_key: str = ""
    # "implicit" writes at the end of the prompt and at every block boundary the
    # response crosses; "explicit" writes only at client breakpoints, and
    # nothing for the tail of the response.
    mode: str = "implicit"
    # Requested lifetime; the node clamps it to its own `max_cache_time`.
    ttl_seconds: int | None = None
    # Message indices marked as write points, into the request's final message
    # list (so after `instructions` and any tool instructions were inserted).
    # Read off any Responses request that carries them - a malformed one is an
    # error whatever the mode - but only explicit mode acts on them.
    breakpoints: list[int] = field(default_factory=list)

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


def _marks_a_breakpoint(item: Any) -> bool:
    """Whether one `input` item ends a stable block.

    The marker rides a content block, but what it names is the end of the
    message that block belongs to: a prefix has to end on a message boundary or
    the chat template's own framing would be cut in half.
    """
    if not isinstance(item, dict):
        return False
    # `content` on a message, `output` on a tool result: the two places a
    # Responses item keeps content blocks.
    blocks = []
    for value in (item.get("content"), item.get("output")):
        blocks.extend(value if isinstance(value, list) else [value])
    return any(_breakpoint_on(block) for block in blocks)


def input_breakpoints(response_input: Any) -> List[int]:
    """Indices of the `input` items marked with a breakpoint, at most four."""
    items = response_input if isinstance(response_input, list) else [response_input]
    marked = [i for i, item in enumerate(items) if _marks_a_breakpoint(item)]
    return marked[:MAX_BREAKPOINTS]


def _reject_instruction_breakpoints(instructions: Any):
    """`instructions` is a string, so it has nowhere to hang a breakpoint.

    A client that sends one meant to cache a long system prompt; say so, rather
    than accepting the request and quietly caching nothing.
    """
    if instructions is None or isinstance(instructions, str):
        return
    blocks = instructions if isinstance(instructions, list) else [instructions]
    if any(_breakpoint_on(block) for block in blocks):
        raise CacheOptionError(
            "prompt_cache_breakpoint is not allowed on instructions; "
            "move the text into an input developer message and mark it there"
        )


def _parse_ttl(raw: Any, parameter: str) -> Optional[int]:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in TTL_VALUES:
        allowed = "', '".join(k for k in TTL_VALUES if k != "in-memory")
        raise CacheOptionError(f"unknown {parameter} '{raw}', expected one of '{allowed}'")
    return TTL_VALUES[raw]


def _parse_mode(options: Dict[str, Any]) -> str:
    raw = options.get("mode")
    if raw is None:
        return "implicit"
    if raw not in MODES:
        allowed = "', '".join(MODES)
        raise CacheOptionError(
            f"unknown prompt_cache_options.mode '{raw}', expected one of '{allowed}'"
        )
    return raw


def parse_cache_options(
    data: Dict[str, Any],
    responses: bool = False
) -> CacheOptions:
    """Read the caching parameters off a request body.

    `authenticated` is whether the server has `api_keys` configured, not whether
    this particular request carried one - an unauthenticated server passes the
    literal `"anon"` as the key for everybody, so identity there rests entirely
    on `prompt_cache_key`.

    `responses` selects the surface: only `/v1/responses` accepts modes, TTLs
    and breakpoints, and only there is an unknown value an error. On chat
    completions those parameters do not exist, so sending them is not a mistake
    the server has any business reporting.
    """
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

    # The older spelling of the same request. `prompt_cache_options.ttl` wins if
    # a client sends both, since it is the one the current API documents.
    retention = _parse_ttl(data.get("prompt_cache_retention"), "prompt_cache_retention")
    if options.ttl_seconds is None and retention is not None:
        options.ttl_seconds = retention

    _reject_instruction_breakpoints(data.get("instructions"))
    options.breakpoints = input_breakpoints(data.get("input"))
    return options


def cached_tokens(job: Any) -> int:
    """What the job reused, or 0 for anything that never got that far.

    Read through `job.caching` (`JobCache`) rather than off the job, and
    defensively: the usage block is built for whatever object the API layer was
    handed, and a job that never reached the cache path still has to report.
    """
    return getattr(getattr(job, "caching", None), "cached_tokens", 0)


def cache_write_tokens(job: Any) -> int:
    """What the job newly committed to the cache, as `cached_tokens`' opposite.

    Counted on the origin only: the other nodes on the pipe store the same
    boundaries under the same IDs, and none of that is observable from here.
    """
    return getattr(getattr(job, "caching", None), "write_tokens", 0)


def input_tokens_details(job: Any) -> Dict[str, int]:
    """`usage.input_tokens_details` for the Responses API."""
    return {
        "cached_tokens": cached_tokens(job),
        "cache_write_tokens": cache_write_tokens(job)
    }


def prompt_tokens_details(job: Any) -> Dict[str, int]:
    """`usage.prompt_tokens_details` for chat completions.

    No `cache_write_tokens` here: OpenAI does not expose it on this endpoint,
    and a field only this server reports is a field clients cannot rely on.
    """
    return {"cached_tokens": cached_tokens(job)}


def responses_usage(job: Any) -> Dict[str, Any]:
    return {
        "input_tokens": job.prompt_tokens,
        "input_tokens_details": input_tokens_details(job),
        "output_tokens": job.current_token,
        "total_tokens": job.prompt_tokens + job.current_token
    }


def chat_usage(job: Any) -> Dict[str, Any]:
    return {
        "prompt_tokens": job.prompt_tokens,
        "prompt_tokens_details": prompt_tokens_details(job),
        "completion_tokens": job.current_token,
        "total_tokens": job.prompt_tokens + job.current_token
    }
