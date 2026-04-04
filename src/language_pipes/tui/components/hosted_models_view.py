from typing import List, Dict
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
    if len(running) == 0:
        lines.append("       Not Running")
    
    pipes: Dict[str, List[ModelStatusInfo]] = { }

    for mi in running:
        if mi.end_model:
            has_ends = True
        if mi.pipe_id == '':
            continue
        if mi.pipe_id not in pipes:
            pipes[mi.pipe_id] = []
        pipes[mi.pipe_id].append(mi)

    pipe_strings = { }
    for key in pipes.keys():
        pipe = pipes[key]
        pipe_string = [" " for _ in range(pipe[0].num_layers - 1)]
        for mi in pipe:
            ch = "X"
            if mi.status == ModelStatus.Running:
                ch = "="
            if mi.status == ModelStatus.Starting:
                ch = "|"
            for i in range(mi.start_layer, mi.end_layer):
                pipe_string[i] = ch
        pipe_strings[key] = ''.join(pipe_string)
    
    for key in pipe_strings.keys():
        lines.append(f"       Pipe {key[:4]} >{pipe_strings[key]}<")

    if has_ends:
        lines.append("       Ends loaded")

    return "\n".join(lines)
