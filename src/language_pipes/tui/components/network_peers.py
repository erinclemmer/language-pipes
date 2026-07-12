from typing import List, Dict, Callable

from language_pipes.content_provider.content_provider import ContentProvider
from ansinout import PressedKey
from language_pipes.distributed_state_network.objects.state_packet import StatePacket
from language_pipes.tui.components.view_pipe import format_pipe_view
from language_pipes.tui.util.text import make_footer_text, make_window_text


class NetworkPeers:
    provider: ContentProvider
    exit_page: Callable
    peers: Dict[str, StatePacket]
    scroll_idx: int

    def __init__(
        self, provider: ContentProvider, exit_page: Callable
    ):
        self.provider = provider
        self.exit_page = exit_page
        self.peers = {}
        self.scroll_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()
        if key == PressedKey.ArrowDown:
            self.scroll_idx = (self.scroll_idx + 1) % len(self.peers.keys())
        if key == PressedKey.ArrowUp:
            self.scroll_idx = (self.scroll_idx - 1) % len(self.peers.keys())

    def get_view(self) -> List[str]:
        self.node_id = self.provider.network_provider.get_network_config().node_id
        self.peers = self.provider.network_provider.get_peers()
        if not self.peers:
            return ["Network Peers:", "", "No peers connected"]
        lines = [
            f"My ID: {self.node_id}",
            "",
            "Network Peers:", 
        ]

        pipes = self.provider.pipe_provider.get_network_pipes()
        assert pipes is not None

        peers = []
        for key in self.peers.keys():
            peer_lines = []
            peer_lines.append(f"{key}")
            endpoint = self.provider.network_provider.get_peer_endpoint(key)
            if endpoint is not None:
                peer_lines.append(f"{endpoint.address}:{endpoint.port}")
            for pipe in pipes:
                if len([s for s in pipe.segments if s.node_id == key]) == 0:
                    continue
                peer_lines.extend(format_pipe_view(pipe))
            peer_lines.append("")
            peers.append(peer_lines)

        lines.extend(make_window_text(peers, self.scroll_idx, 15))

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrow U/D: Scroll Page", "Esc: Menu"])
