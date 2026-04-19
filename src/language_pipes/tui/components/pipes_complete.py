from typing import List, Callable

from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.content_loader import ContentLoader
from language_pipes.content_provider.provider_calls import ProviderCall
from language_pipes.tui.components.view_pipe import format_pipe_view

class PipesComplete:
    provider: ContentProvider
    network_pipes: List[MetaPipe]

    def __init__(
        self,
        provider: ContentProvider,
        exit_page: Callable,
        is_focused: Callable,
    ):
        self.provider = provider
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        self.network_pipes = self.provider.call_provider(ProviderCall.get_network_pipes)
        if self.network_pipes is None:
            return ["Network Not Connected", "Connect to the network to view available pipes"]

        pipes_to_show = []
        for pipe in self.network_pipes:
            if pipe.is_complete(0):
               pipes_to_show.append(pipe)
                
        if len(pipes_to_show) == 0:
            return ["No Pipes unconnected pipes found on network"]
        
        lines = []
        for pipe in pipes_to_show:
            lines.extend(format_pipe_view(pipe))

        return lines

    def get_footer(self) -> str:
        return "Esc: Back"
