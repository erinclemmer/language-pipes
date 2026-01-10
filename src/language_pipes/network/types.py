from typing import Any, List, Optional, Protocol


class NetworkConnection(Protocol):
    address: str


class StateNetworkNode(Protocol):
    logger: Any
    shutting_down: bool
    node_id: str
    port: int

    def read_data(self, node_id: str, key: str) -> Optional[str]:
        ...

    def update_data(self, key: str, value: str) -> None:
        ...

    def peers(self) -> List[str]:
        ...

    def connection_from_node(self, node_id: str) -> NetworkConnection:
        ...


class StateNetworkServer(Protocol):
    node: StateNetworkNode
    logger: Any

    def stop(self) -> None:
        ...
