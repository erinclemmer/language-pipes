from typing import List, Callable

from ansinout import PressedKey
from language_pipes.tui.components.view_pipe import format_pipe_view
from language_pipes.content_provider.content_provider import ContentProvider

class PipesIncomplete:
    provider: ContentProvider

    def __init__(
        self,
        provider: ContentProvider,
        exit_page: Callable,
    ):
        self.provider = provider
        self.exit_page = exit_page

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        network_pipes = self.provider.pipe_provider.get_network_pipes()
        if network_pipes is None:
            return ["Network Not Connected", "Connect to the network to view available pipes"]
        
        pipes_to_show = []
        for pipe in network_pipes:
            if not pipe.is_complete(0):
                pipes_to_show.append(pipe)

        if len(pipes_to_show) == 0:
            return ["No Pipes unconnected pipes found on network"]
        
        lines = []
        for pipe in pipes_to_show:
            num_local_layers = self.provider.model_provider.get_num_local_layers_for(pipe.model_id)
            lines.extend(format_pipe_view(pipe, num_local_layers))

        return lines

    def get_footer(self) -> str:
        return "Esc: Menu"
