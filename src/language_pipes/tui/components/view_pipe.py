from typing import List

from language_pipes.config import DEFAULT_NUM_LOCAL_LAYERS
from language_pipes.pipes.meta_pipe import MetaPipe

def format_pipe_view(
    pipe: MetaPipe,
    num_local_layers: int = DEFAULT_NUM_LOCAL_LAYERS
) -> List[str]:
    lines = [
        pipe.model_id
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

    PIPE_LEN = 24

    norm_pipe = ["X" for _ in range(0, PIPE_LEN)]
    layers_per_char =  num_layers / float(PIPE_LEN)
    for i in range(0, PIPE_LEN):
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
    
    norm_pipe = "".join(norm_pipe)
    node_id_string = ", ".join(node_ids)
    completed_string = "(=)" if pipe.is_complete(num_local_layers) else "(X)"
    lines.extend([
        f"ID {pipe.pipe_id[:4]} >{norm_pipe}< {completed_string}",
        f"{len(node_ids)} Node(s): " + ((node_id_string[:20] + "...") if len(node_id_string) > 20 else node_id_string)
    ])

    return lines