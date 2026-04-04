from typing import List, Callable

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.components.view_pipe import format_pipe_view


# TODO: Split this into PipesComplete and PipesIncomplete, allow joining on PipesIncomplete
class PipesAvailable:
    loader: ContentLoader
    pipes_available: List[MetaPipe]

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
        self.pipes_available = self.loader.call_provider(ProviderCall.get_available_pipes)
        if self.pipes_available is None:
            return ["Network Not Connected", "Connect to the network to view available pipes"]
        if len(self.pipes_available) == 0:
            return ["No Pipes unconnected pipes found on network"]
        
        lines = []
        for pipe in self.pipes_available:
            lines.extend(format_pipe_view(pipe))

        return lines

    def get_footer(self) -> str:
        return "Esc: Back"
