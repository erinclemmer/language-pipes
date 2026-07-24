from typing import List, Callable

from ansinout import PressedKey
from language_pipes.tui.components.view_pipe import format_pipe_view
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.util.text import make_window_text

class PipesConnected:
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
        pipes_connected = self.provider.pipe_provider.get_connected_pipes()
        if pipes_connected is None:
            return ["Network Not Connected", "Connect to the network to view connected pipes"]
        if len(pipes_connected) == 0:
            return ["No Pipes Connected", "Host a model to connect to a pipe"]
        
        entries = []
        for pipe in pipes_connected:
            num_local_layers = self.provider.model_provider.get_num_local_layers_for(pipe.model_id)
            line = format_pipe_view(pipe, num_local_layers)
            line.append("")
            entries.append(line)

        lines = make_window_text(entries, 0, 17)

        return lines

    def get_footer(self) -> str:
        return "Esc: Menu"
