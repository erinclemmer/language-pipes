from typing import Any


def format_model_line(model: Any, selected: bool = False) -> str:
    l_cursor = "|>" if selected else "  "
    r_cursor = "<|" if selected else "  "
    ends_string = "+ ends" if getattr(model, "load_ends", False) else ""
    return (
        f"{l_cursor} {getattr(model, 'model_id', '')} "
        f"{getattr(model, 'max_memory', '')}GB {ends_string} "
        f"{getattr(model, 'device', '')} {r_cursor}"
    )
