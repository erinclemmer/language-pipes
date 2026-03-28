from typing import List, Dict, Callable

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.distributed_state_network.objects.state_packet import StatePacket

class NetworkPeers:
    loader: ContentLoader
    exit_page: Callable
    is_focused: Callable
    peers: Dict[str, StatePacket]

    def __init__(self, loader: ContentLoader, exit_page: Callable, is_focused: Callable):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.peers = { }

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        self.peers = self.loader.call_provider(ProviderCall.list_peers)
        lines = ["Network Peers:", ""]
        for key in self.peers.keys():
            lines.append(f"- {key}")

        return lines
    
    def get_footer(self) -> str:
        return ""