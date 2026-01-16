from typing import Any, List, Optional, Protocol, Callable

class StateNetworkServer(Protocol):
    logger: Any
    node_id: str
    receive_cb: Callable[[bytes], None]
    
    def read_data(self, node_id: str, key: str) -> Optional[str]:
        ...

    def update_data(self, key: str, value: str) -> None:
        ...

    def peers(self) -> List[str]:
        ...

    def stop(self) -> None:
        ...

    def is_shut_down(self) -> bool:
        ...

    def send_to_node(self, node_id: str):
        ...