from typing import List, Callable

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.content_loader import ContentLoader
from language_pipes.content_provider.provider_calls import ProviderCall
from language_pipes.tui.components.view_pipe import format_pipe_view

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
            lines.extend(format_pipe_view(pipe))

        return lines

    def get_footer(self) -> str:
        return "Esc: Back"
