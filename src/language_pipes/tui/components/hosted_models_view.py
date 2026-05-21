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

    norm_pipe_strings = { }
    for key in pipes.keys():
        pipe = pipes[key]
        pipe_string = ["X" for _ in range(0, 28)]
        layers_per_char =  pipe[0].num_layers / 28.0
        for i in range(0, 28):
            start = int(layers_per_char * i)
            end = int(layers_per_char * (i + 1))
            loading = False
            loaded = True
            for c in pipe_strings[key][start:end]:
                if c == "|":
                    loading = True
                    break
                if c == "X":
                    loaded = False
                    break
            
            if loading:
                pipe_string[i] = "|"
            elif loaded:
                pipe_string[i] = "="
        norm_pipe_strings[key] = "".join(pipe_string)
    
    lines = []
    for key in pipe_strings.keys():
        lines.append(f"Pipe {key[:4]} >{norm_pipe_strings[key]}<")

    return lines

def format_model_line(
    model: ModelToLoad,
    selected: bool = False,
    running: List[ModelStatusInfo] = []
) -> List[str]:
    l_cursor = "|>" if selected else "  "
    r_cursor = "<|" if selected else "  "
    lines = [
        f"{l_cursor} {model.model_id} on {model.device} {r_cursor} ",
        f"       Max Memory: {model.memory}GB"
    ]
    has_ends = len([m for m in running if m.end_model]) > 0
    
    if len(running) == 0:
        lines.append("       Not Running")
    
    pipe_strings = format_pipe_strings(running)

    for pipe in pipe_strings:
        lines.append(f"       {pipe}")

    if has_ends:
        lines.append("       Ends loaded")

    return lines
