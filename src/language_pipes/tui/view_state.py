"""
ViewState helpers: building and formatting view-state dicts for each section.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple


def build_view_state(
    state: str,
    summary: str,
    details: List[str],
    hint: str,
    level: str,
) -> Dict[str, Any]:
    normalized_details = [str(line) for line in details if str(line).strip() != ""]
    return {
        "state": state,
        "summary": summary,
        "details": normalized_details,
        "hint": hint,
        "level": level,
    }


def error_view_state(summary: str, hint: str) -> Dict[str, Any]:
    return build_view_state("error", summary, [], hint, "error")


def empty_view_state(summary: str, hint: str) -> Dict[str, Any]:
    return build_view_state("empty", summary, [], hint, "info")


def _dict_preview(payload: Dict[str, Any], limit: int = 5) -> List[str]:
    preview: List[str] = []
    for idx, key in enumerate(payload.keys()):
        if idx >= limit:
            break
        preview.append(f"- {key}: {payload[key]}")
    if len(payload) > limit:
        preview.append(f"- ... ({len(payload) - limit} more)")
    return preview


def _item_name(item: Dict[str, Any], fallback_idx: int) -> str:
    for key in ("name", "id", "model", "peer", "route", "event"):
        value = item.get(key)
        if value:
            return str(value)
    return f"item-{fallback_idx + 1}"


def format_network(tab: str, section: str, payload: Any) -> Dict[str, Any]:
    if section == "Status":
        if not isinstance(payload, dict):
            return error_view_state(
                "Malformed network status payload.",
                "Next: Confirm get_network_status returns a dict, then press r.",
            )
        if len(payload) == 0:
            return empty_view_state(
                "No network status data yet.",
                "Next: Start networking components and press r to refresh.",
            )
        health = payload.get("status") or payload.get("state") or payload.get("health") or "unknown"
        details = _dict_preview(payload)
        return build_view_state(
            "ok",
            f"Network health: {health}",
            details,
            "Next: Use Network -> Peers for connected node details.",
            "info",
        )

    if section == "Peers":
        if not isinstance(payload, list):
            return error_view_state(
                "Malformed peers payload.",
                "Next: Confirm list_peers returns a list of dict items, then press r.",
            )
        if len(payload) == 0:
            return empty_view_state(
                "No peers connected yet.",
                "Next: Check bootstrap connectivity and press r.",
            )
        lines: List[str] = []
        for i, peer in enumerate(payload[:5]):
            if isinstance(peer, dict):
                name = _item_name(peer, i)
                addr = peer.get("address") or peer.get("host") or "unknown"
                state = peer.get("state") or peer.get("status") or "unknown"
                lines.append(f"- {name} @ {addr} ({state})")
            else:
                lines.append(f"- {peer}")
        if len(payload) > 5:
            lines.append(f"- ... ({len(payload) - 5} more peers)")
        return build_view_state(
            "ok",
            f"Connected peers: {len(payload)}",
            lines,
            "Next: Press r to re-check discovery/connectivity.",
            "info",
        )

    return format_unknown(tab, section, payload)


def format_models(_: str, section: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        return error_view_state(
            "Malformed models payload.",
            "Next: Confirm list_models returns a list of dict items, then press r.",
        )
    if len(payload) == 0:
        return empty_view_state(
            "No model data available.",
            "Next: Ensure model manager is initialized and press r.",
        )
    lines: List[str] = []
    for i, model in enumerate(payload[:6]):
        if isinstance(model, dict):
            name = _item_name(model, i)
            state = model.get("status")
            if section == "Cache":
                cache_size = model.get("cache_size") or model.get("cache")
                if cache_size is not None:
                    lines.append(f"- {name}: cache={cache_size}")
                else:
                    lines.append(f"- {name}: cache info unavailable")
            elif state is not None:
                lines.append(f"- {name}: {state}")
            else:
                lines.append(f"- {name}")
        else:
            lines.append(f"- {model}")
    if len(payload) > 6:
        lines.append(f"- ... ({len(payload) - 6} more models)")
    section_label = {
        "Installed": "Installed models",
        "Download": "Model download candidates",
        "Cache": "Model cache summary",
    }.get(section, "Model summary")
    return build_view_state(
        "ok",
        f"{section_label}: {len(payload)}",
        lines,
        "Next: Press r to refresh model inventory.",
        "info",
    )


def format_pipes(_: str, section: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return error_view_state(
            "Malformed pipe health payload.",
            "Next: Confirm get_pipe_health returns a dict, then press r.",
        )
    if len(payload) == 0:
        return empty_view_state(
            "No pipe health data yet.",
            "Next: Start routing/services and press r.",
        )
    summary_health = payload.get("health") or payload.get("status") or payload.get("state") or "unknown"
    details = _dict_preview(payload)
    return build_view_state(
        "ok",
        f"{section} health: {summary_health}",
        details,
        "Next: Verify route configuration if health is degraded.",
        "info",
    )


def format_jobs(_: str, section: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        return error_view_state(
            "Malformed jobs payload.",
            "Next: Confirm list_jobs returns a list of dict items, then press r.",
        )
    if len(payload) == 0:
        return empty_view_state(
            "No job data in this view.",
            "Next: Run workloads and press r.",
        )
    if section == "Stats":
        by_state: Dict[str, int] = {}
        for item in payload:
            if isinstance(item, dict):
                s = str(item.get("state") or "unknown")
            else:
                s = "unknown"
            by_state[s] = by_state.get(s, 0) + 1
        lines = [f"- {s}: {count}" for s, count in sorted(by_state.items())]
        return build_view_state(
            "ok",
            f"Tracked jobs: {len(payload)}",
            lines,
            "Next: Use Queue/History sections for item-level detail.",
            "info",
        )
    lines: List[str] = []
    for i, job in enumerate(payload[:6]):
        if isinstance(job, dict):
            job_id = _item_name(job, i)
            state = job.get("state") or "unknown"
            progress = job.get("progress")
            if progress is None:
                lines.append(f"- {job_id} ({state})")
            else:
                lines.append(f"- {job_id} ({state}, {progress})")
        else:
            lines.append(f"- {job}")
    if len(payload) > 6:
        lines.append(f"- ... ({len(payload) - 6} more jobs)")
    return build_view_state(
        "ok",
        f"{section} jobs: {len(payload)}",
        lines,
        "Next: Press r to sync the latest job state.",
        "info",
    )


def format_activity(_: str, section: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        return error_view_state(
            "Malformed activity payload.",
            "Next: Confirm list_activity returns a list, then press r.",
        )
    if len(payload) == 0:
        return empty_view_state(
            "No activity captured yet.",
            "Next: Trigger workload/network activity and press r.",
        )
    lines: List[str] = []
    for i, item in enumerate(payload[:6]):
        if isinstance(item, dict):
            name = _item_name(item, i)
            level = item.get("level") or item.get("type") or "info"
            message = item.get("message") or item.get("summary")
            if message:
                lines.append(f"- [{level}] {name}: {message}")
            else:
                lines.append(f"- [{level}] {name}")
        else:
            lines.append(f"- {item}")
    if len(payload) > 6:
        lines.append(f"- ... ({len(payload) - 6} more events)")
    return build_view_state(
        "ok",
        f"{section} entries: {len(payload)}",
        lines,
        "Next: Press r to pull the latest activity snapshot.",
        "info",
    )


def format_unknown(tab: str, section: str, _: Any) -> Dict[str, Any]:
    return build_view_state(
        "placeholder",
        f"No provider mapping for {tab} -> {section}.",
        [],
        "Next: Select another section or wire a provider for this view.",
        "warning",
    )


# Maps (tab, section) -> (provider_name, kwargs, formatter)
_FORMATTER_TYPE = Callable[[str, str, Any], Dict[str, Any]]


def section_provider_spec(
    tab: str, section: str
) -> Tuple[Optional[str], Dict[str, Any], _FORMATTER_TYPE]:
    if tab == "Network" and section == "Status":
        return "get_network_status", {}, format_network
    if tab == "Network" and section == "Peers":
        return "list_peers", {}, format_network

    if tab == "Models":
        return "list_models", {}, format_models

    if tab == "Pipes":
        return "get_pipe_health", {}, format_pipes

    if tab == "Jobs":
        state_map = {
            "Queue": "queued",
            "History": "completed",
            "Stats": None,
        }
        return "list_jobs", {"state": state_map.get(section)}, format_jobs

    if tab == "Activity":
        level_map = {
            "Logs": "info",
            "Events": "event",
            "Metrics": "metrics",
        }
        return "list_activity", {"level": level_map.get(section)}, format_activity

    return None, {}, format_unknown
