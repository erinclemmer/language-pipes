from typing import Callable

from language_pipes.network.config import NetworkConfig
from language_pipes.network.types import StateNetworkServer


def start_state_network(
    config: NetworkConfig,
    update_callback: Callable,
    disconnect_callback: Callable,
) -> StateNetworkServer:
    if config.provider == "dsn":
        from language_pipes.network.dsn import start_dsn_server

        return start_dsn_server(
            settings=config.settings,
            update_callback=update_callback,
            disconnect_callback=disconnect_callback,
        )
    raise ValueError(f"Unknown network provider '{config.provider}'")
