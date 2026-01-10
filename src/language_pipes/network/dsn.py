from typing import Any, Callable, Dict

from distributed_state_network import DSNodeConfig, DSNodeServer

from language_pipes.network.types import StateNetworkServer


class DsnNetworkServer:
    def __init__(self, server: DSNodeServer):
        self._server = server
        self.node = server.node
        self.logger = server.logger

    def stop(self) -> None:
        self._server.stop()


def start_dsn_server(
    settings: Dict[str, Any],
    update_callback: Callable,
    disconnect_callback: Callable,
) -> StateNetworkServer:
    config = DSNodeConfig.from_dict(settings)
    server = DSNodeServer.start(
        config=config,
        update_callback=update_callback,
        disconnect_callback=disconnect_callback,
    )
    return DsnNetworkServer(server)
