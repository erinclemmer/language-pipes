from typing import List, Dict
from language_pipes.content_provider.model_provider import ModelStatusInfo, ModelToLoad, ModelStatus

def format_pipe_strings(running: List[ModelStatusInfo]) -> List[str]:
    pipes: Dict[str, List[ModelStatusInfo]] = { }

    for mi in running:
        if mi.pipe_id == '':
            continue
        if mi.pipe_id not in pipes:
            pipes[mi.pipe_id] = []
        pipes[mi.pipe_id].append(mi)

    pipe_strings = { }
    for key in pipes.keys():
        pipe = pipes[key]
        pipe_string = ["X" for _ in range(pipe[0].num_layers - 1)]
        for mi in pipe:
            ch = "X"
            if mi.status == ModelStatus.Running:
                ch = "="
            if mi.status == ModelStatus.Starting:
                ch = "|"
            for i in range(mi.start_layer, mi.end_layer):
                pipe_string[i] = ch
        pipe_strings[key] = ''.join(pipe_string)

    lines = []
    for key in pipe_strings.keys():
        lines.append(f"Pipe {key[:4]} >{pipe_strings[key]}<")
    return lines

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
    has_ends = len([m for m in running if m.end_model]) > 0
    
    if len(running) == 0:
        lines.append("       Not Running")
    
    pipe_strings = format_pipe_strings(running)

    for pipe in pipe_strings:
        lines.append(f"       {pipe}")

    if has_ends:
        lines.append("       Ends loaded")

    return "\n".join(lines)
