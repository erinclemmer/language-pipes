from typing import List, Callable

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.util.kb_utils import PressedKey


class PipesConnected:
    loader: ContentLoader
    pipes_connected: List[MetaPipe]

    def __init__(
        self,
        loader: ContentLoader,
        exit_page: Callable,
        is_focused: Callable,
    ):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        self.pipes_connected = self.loader.call_provider(ProviderCall.get_pipes_connected)
        if self.pipes_connected is None:
            return ["Network Not Connected", "Connect to the network to view connected pipes"]
        if len(self.pipes_connected) == 0:
            return ["No Pipes Connected", "Host a model to connect to a pipe"]
        
        lines = []
        for pipe in self.pipes_connected:
            lines.extend([
                f"Pipe ID: {pipe.pipe_id[:8]}",
                f"Model ID: {pipe.model_id}"
            ])
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
                f"{len(node_ids)} node(s) connected",
                "Nodes: " + ((node_id_string[:40] + "...") if len(node_id_string) > 40 else node_id_string )
            ])

        return lines

    def get_footer(self) -> str:
        return "Esc: Back"
