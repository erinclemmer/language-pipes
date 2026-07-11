from typing import List, Dict
from language_pipes.content_provider.model_provider import ModelStatusInfo, ModelToLoad, ModelStatus
from language_pipes.tui.util.text import make_selectable_text
from language_pipes.util.config import is_8_bit_mode

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
        pipe = pipes[key]
        ram_used = sum([m.ram_used for m in pipe]) / 1024**3
        if is_8_bit_mode():
            ram_used /= 2.0
        lines.append(f"Pipe {key[:4]} >{norm_pipe_strings[key]}< ({ram_used:.2f}GB)")

    return lines

def format_model_line(
    model: ModelToLoad,
    selected: bool = False,
    running: List[ModelStatusInfo] = []
) -> List[str]:
    line = make_selectable_text(f"{model.model_id} on {model.device}", selected)
    lines = [
        line,
        f"       Max Memory: {model.memory}GB"
    ]
    if len(running) == 0:
        lines.append("       Not Running")
    
    pipe_strings = format_pipe_strings(running)

    for pipe in pipe_strings:
        lines.append(f"       {pipe}")

    return lines
