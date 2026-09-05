"""Prompt-caching request options and usage reporting for the OpenAI surface.

Kept out of `oai.py` because the parsing rule that matters here is a privacy
rule, not a serialization one: on a server with no `api_keys` configured every
caller shares one identity, so `prompt_cache_key` is the only thing separating
them and a request without one must not touch the cache at all.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheOptions:
    # Partitions the cache within an API key. On an unauthenticated server it is
    # also the only thing partitioning at all, which is why an absent value
    # disables caching there.
    prompt_cache_key: str = ""
    # "implicit" writes at the end of the prompt and at the end of the response;
    # "explicit" writes only at client breakpoints. Phase 1 only ever sees
    # "implicit" - `prompt_cache_options` is not parsed yet.
    mode: str = "implicit"
    # Requested lifetime; the node clamps it to its own `max_cache_time`.
    ttl_seconds: Optional[int] = None
    # Message indices marked as write points. Always empty until explicit mode
    # lands.
    breakpoints: List[int] = field(default_factory=list)
    enabled: bool = False


def parse_cache_options(data: Dict[str, Any], authenticated: bool) -> CacheOptions:
    """Read the caching parameters off a request body.

    `authenticated` is whether the server has `api_keys` configured, not whether
    this particular request carried one - an unauthenticated server passes the
    literal `"anon"` as the key for everybody, so identity there rests entirely
    on `prompt_cache_key`.

    Unknown values are not rejected: Phase 1 must not start 400ing requests the
    server accepts today. Validation arrives with the rest of the surface.
    """
    raw_key = data.get("prompt_cache_key")
    prompt_cache_key = str(raw_key) if isinstance(raw_key, str) else ""
    return CacheOptions(
        prompt_cache_key=prompt_cache_key,
        enabled=authenticated or prompt_cache_key != ""
    )


def cached_tokens(job: Any) -> int:
    """What the job reused, or 0 for anything that never got that far.

    Read through `job.caching` (`JobCache`) rather than off the job, and
    defensively: the usage block is built for whatever object the API layer was
    handed, and a job that never reached the cache path still has to report.
    """
    return getattr(getattr(job, "caching", None), "cached_tokens", 0)


def input_tokens_details(job: Any) -> Dict[str, int]:
    """`usage.input_tokens_details` for the Responses API."""
    return {"cached_tokens": cached_tokens(job)}


def prompt_tokens_details(job: Any) -> Dict[str, int]:
    """`usage.prompt_tokens_details` for chat completions."""
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
