from typing import List
from language_pipes.tui.content_provider.model_provider import ModelStatusInfo, ModelToLoad


def format_model_line(
    model: ModelToLoad,
    selected: bool = False,
    running: List[ModelStatusInfo] = []
) -> str:
    l_cursor = "|>" if selected else "  "
    r_cursor = "<|" if selected else "  "
    ends_string = "Yes" if model.load_ends else "No"
    lines = [
        f"{l_cursor} {model.model_id} {r_cursor} ",
        f"       Max Memory: {model.max_memory}GB",
        f"       Load Ends: {ends_string}"
    ]
    for mi in running:
        lines.extend([
            f"           Layers: {mi.layers_loaded} {mi.status.value}"
        ])
    return "\n".join(lines)
