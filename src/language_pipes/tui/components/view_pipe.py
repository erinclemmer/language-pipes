from typing import List

from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.pipes.meta_pipe import MetaPipe

def format_pipe_view(
    pipe: MetaPipe
) -> List[str]:
    lines = [
        f"Pipe ID: {pipe.pipe_id[:8]}",
        f"Model ID: {pipe.model_id}"
    ]
    num_layers = pipe.num_layers()
    pipe_list = ["X" for _ in range(num_layers - 1)]
    node_ids = set()
    for segment in pipe.segments:
        if segment.node_id not in node_ids:
            node_ids.add(segment.node_id)
        ch = "|"
        if segment.loaded:
            ch =  "="
        for i in range(segment.start_layer, min(segment.end_layer + 1, num_layers - 1)):
            pipe_list[i] = ch

    norm_pipe = ["X" for _ in range(0, 28)]
    layers_per_char =  num_layers / 28.0
    for i in range(0, 28):
        start = int(layers_per_char * i)
        end = int(layers_per_char * (i + 1))
        loading = False
        loaded = True
        for c in pipe_list[start:end]:
            if c == "|":
                loading = True
                break
            if c == "X":
                loaded = False
                break
        
        if loading:
            norm_pipe[i] = "|"
        elif loaded:
            norm_pipe[i] = "="
    
    norm_pipe = "".join(pipe_list)
    node_id_string = ", ".join(node_ids)
    lines.extend([
        f">{norm_pipe}<",
        "Pipe Complete" if pipe.is_complete(ModelProvider.get_num_local_layers()) else "Pipe Incomplete",
        f"{len(node_ids)} node(s) connected",
        "Nodes: " + ((node_id_string[:40] + "...") if len(node_id_string) > 40 else node_id_string )
    ])

    return lines