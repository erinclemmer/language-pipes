from typing import Any, Callable, Dict

from distributed_state_network import DSNode, DSNodeConfig, DSNodeServer

from language_pipes.network.types import StateNetworkNode, StateNetworkServer


class DsnNetworkNode:
    def __init__(self, node: DSNode):
        self._node = node

    @property
    def logger(self):
        return self._node.logger

    @property
    def shutting_down(self) -> bool:
        return self._node.shutting_down

    @property
    def node_id(self) -> str:
        return self._node.config.node_id

    @property
    def port(self) -> int:
        return self._node.config.port

    def read_data(self, node_id: str, key: str) -> str | None:
        return self._node.read_data(node_id, key)

    def update_data(self, key: str, value: str) -> None:
        self._node.update_data(key, value)

    def peers(self):
        return self._node.peers()

    def connection_from_node(self, node_id: str):
        return self._node.connection_from_node(node_id)


class DsnNetworkServer:
    def __init__(self, server: DSNodeServer):
        self._server = server
        self.node: StateNetworkNode = DsnNetworkNode(server.node)
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
