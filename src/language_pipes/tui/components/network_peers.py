from typing import List, Dict, Callable

from language_pipes.content_provider.content_provider import ContentProvider
from ansinout import PressedKey
from language_pipes.distributed_state_network.objects.state_packet import StatePacket


class NetworkPeers:
    provider: ContentProvider
    exit_page: Callable
    is_focused: Callable
    peers: Dict[str, StatePacket]

    def __init__(
        self, provider: ContentProvider, exit_page: Callable, is_focused: Callable
    ):
        self.provider = provider
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.peers = {}

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        self.peers = self.provider.network_provider.get_peers()
        if not self.peers:
            return ["Network Peers:", "", "No peers connected"]
        lines = ["Network Peers:", ""]
        for key in self.peers.keys():
            lines.append(f"- {key}")

        return lines

    def get_footer(self) -> str:
        return ""
