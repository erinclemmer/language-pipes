## DSNode

Main node class that handles all distributed state network operations including peer management, state synchronization, and encrypted communication.

```python
from distributed_state_network import DSNode
```

### Class Definition
```python
class DSNode:
    config: DSNodeConfig
    address_book: Dict[str, Endpoint]
    node_states: Dict[str, StatePacket]
    shutting_down: bool = False
```

### Constructor

```python
node = DSNode(config, "0.0.3", disconnect_callback=on_disconnect, update_callback=on_update)
```

**Parameters:**
- `config` (`DSNodeConfig`): Node configuration
- `version` (`str`): Protocol version string for compatibility checking
- `disconnect_callback` (`Optional[Callable]`): Callback function invoked when a peer disconnects
- `update_callback` (`Optional[Callable]`): Callback function invoked when the state of node updates

### Key Methods

### `bootstrap(con: Endpoint) -> None`
Connects to a bootstrap node to join the network.

**Parameters:**
- `con` (`Endpoint`): Connection endpoint of the bootstrap node

**Raises:**
- `Exception`: If connection to bootstrap node fails

### `update_data(key: str, val: str) -> None`
Updates a key-value pair in the node's state and broadcasts the update to all peers.
```python
node.update_data("status", "active")
```

**Parameters:**
- `key` (`str`): State key to update
- `val` (`str`): New value for the key

### `read_data(node_id: str, key: str) -> Optional[str]`
Reads a value from a specific node's state.

**Parameters:**
- `node_id` (`str`): ID of the node to read from
- `key` (`str`): Key to retrieve

**Returns:**
- `Optional[str]`: Value if exists, None otherwise

### `peers() -> List[str]`
Returns a list of all connected peer node IDs.

**Returns:**
- `List[str]`: List of node IDs