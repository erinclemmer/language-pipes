from typing import List

from language_pipes.pipes.meta_pipe import MetaPipe

def format_pipe_view(
    pipe: MetaPipe
) -> List[str]:
    lines = [
        f"Pipe ID: {pipe.pipe_id[:8]}",
        f"Model ID: {pipe.model_id}"
    ]
    pipe_list = ["X" for _ in range(pipe.num_layers() - 1)]
    node_ids = set()
    for segment in pipe.segments:
        if segment.node_id not in node_ids:
            node_ids.add(segment.node_id)
        ch = "|"
        if segment.loaded:
            ch =  "="
        for i in range(segment.start_layer, segment.end_layer):
            pipe_list[i] = ch
    pipe_string = "".join(pipe_list)
    node_id_string = ", ".join(node_ids)
    lines.extend([
        f">{pipe_string}<",
        "Pipe Complete" if pipe.is_complete(0) else "Pipe Incomplete",
        f"{len(node_ids)} node(s) connected",
        "Nodes: " + ((node_id_string[:40] + "...") if len(node_id_string) > 40 else node_id_string )
    ])

    return lines