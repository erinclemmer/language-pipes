from typing import Any, Optional


def format_model_line(
    model: Any,
    selected: bool = False,
    status: str = "Unloaded",
    layers_loaded: Optional[str] = None,
) -> str:
    l_cursor = "|>" if selected else "  "
    r_cursor = "<|" if selected else "  "
    ends_string = "+ ends" if getattr(model, "load_ends", False) else ""
    layers_string = f" layers:{layers_loaded}" if layers_loaded else ""
    return (
        f"{l_cursor} {getattr(model, 'model_id', '')} "
        f"{getattr(model, 'max_memory', '')}GB {ends_string} "
        f"{getattr(model, 'device', '')} [{status}{layers_string}] {r_cursor}"
    )
