from typing import List
from language_pipes.tui.content_provider.model_provider import ModelStatusInfo, ModelToLoad, ModelStatus


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
    has_ends = False
    layers = []
    if len(running) == 0:
        lines.append("       Not Running")
    else:
        layers = [" " for _ in range(running[0].num_layers - 1)]
    
    for mi in running:
        if mi.end_model:
            has_ends = True
            continue
        ch = "X"
        if mi.status == ModelStatus.Running:
            ch = "="
        if mi.status == ModelStatus.Starting:
            ch = "|"

        for i in range(mi.start_layer, mi.end_layer):
            layers[i] = ch
    
    if has_ends:
        lines.append("       Ends loaded")

    if len(layers) > 0:
        lines.append("       Running >" + "".join(layers) + "<")

    return "\n".join(lines)
