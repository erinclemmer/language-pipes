## DSNodeServer

Flask server wrapper for DSNode that handles incoming network requests.

```python
from distributed_state_network import DSNodeServer
```

### Class Definition
```python
class DSNodeServer:
    config: DSNodeConfig
    node: DSNode
    app: Flask
    running: bool
    thread: threading.Thread
```

### Constructor

**Parameters:**
- `config` (`DSNodeConfig`): Node configuration
- `disconnect_callback` (`Optional[Callable]`): Callback for disconnect events
- `update_callback` (`Optional[Callable]`): Callback for state update events
- `receive_callback` (`Optional[Callable]`): Callback for data transfer from one node to another

### Static Methods

### `Start() -> DSNodeServer`
Creates and starts a new DSNodeServer instance with a flask server

```python
server = DSNodeServer.start(config)
```

**Parameters:**
- `config` (`DSNodeConfig`): Node configuration
- `disconnect_callback` (`Optional[Callable]`): Callback for disconnect events (no parameters)
- `update_callback` (`Optional[Callable]`): Callback for state update events (no parameters)
- `receive_callback` (`Optional[Callable]`): Callback for data transfer between nodes

**Returns:**
- `DSNodeServer`: Running server instance

**Example with bootstrap:**
```python
# Bootstrap node (first node in network)
bootstrap_config = DSNodeConfig(
    node_id="bootstrap",
    port=8000,
    bootstrap_nodes=[]
)
bootstrap = DSNodeServer.start(bootstrap_config)

# Connector node (joins existing network)
connector_config = DSNodeConfig(
    node_id="connector",
    port=8001,
    bootstrap_nodes=[Endpoint("127.0.0.1", 8000)]
)
connector = DSNodeServer.start(connector_config)
```

### `generate_key() -> str`
Generates a new hexadecimal encoded AES key for network encryption. All nodes in the same network must share the same AES key.

**Parameters:**
- None

**Example:**
```python
DSNodeServer.generate_key()
```

### Instance Methods

### `stop() -> None`
Gracefully shuts down the server and cleans up resources.

**Example:**
```python
server.stop()
```

### `update_data() -> None`
Updates a key-value pair in the node's state and broadcasts the update to all peers.
```python
node.update_data("status", "active")
```

**Parameters:**
- `key` (`str`): State key to update
- `val` (`str`): New value for the key

### `read_data() -> Optional[str]`
Reads a value from a specific node's state.

**Parameters:**
- `node_id` (`str`): ID of the node to read from
- `key` (`str`): Key to retrieve

**Returns:**
- `Optional[str]`: Value if exists, None otherwise

### `peers() -> List[str]`
Returns a list of all connected peer node IDs.

**Parameters:**
- None

**Returns:**
- `List[str]`: List of node IDs

### `send_to_node() -> None`
Sends data to another node
```python
node.send_to_node('node-1', b'foo bar')
```

**Parameters:**
- `node_id` (`str`): id of node to send data to
- `data` (`bytes`): data to send to node

**Returns:**
- None

### `is_shut_down() -> bool`
Returns whether the server is currently shut down or not

```python
node.is_shut_down()
```

**Parameters:**
- None

**Returns:**
- bool denoting that the server is shutdown if true

### `node_id() -> str`
Return the node_id of this node instance

```python
node.node_id()
```

**Parameters:**
- None

**Returns:**
- string denoting the node id of the current node

### `set_receive_cb() -> None`
Set the receive callback, this function will be called whenever the node receives a data packet.
```python
def receive():
    pass
node.set_receive_cb(receive)
```

**Parameters:**
- `cb` (`Callable[[bytes], None]`) a function that will be given the bytes of the data packet

**Returns:**
- None

### `set_update_cb() -> None`
Set the update callback, this function will be called whenever a new update is made on the server.
```python
def update_cb():
    pass
node.set_update_cb(update_cb)
```

**Parameters:**
- `cb` (`Callable[[], None]`) a function that receives no arguments

**Returns:**
- None

### `set_disconnect_cb() -> None`
Set the disconnect callback, this function will be run whenever a node is disconnected from the network.
```python
def disconnect():
    pass
node.set_disconnect_cb(disconnect)
```

**Parameters:**
- `cb` (`Callable[[], None]`) a function that takes no arguments

**Returns:**
- None

## Network Protocol

The server uses UDP sockets with the following characteristics:

- **Encryption**: All packets can be encrypted with AES
- **Authentication**: ECDSA signatures for message verification

## Message Types

The server handles four types of messages:

1. **HELLO (1)**: Node introduction and public key exchange
2. **PEERS (2)**: Request/response for peer list
3. **UPDATE (3)**: State synchronization
4. **PING (4)**: Connection health check
5. **DATA (5)**: Data transfer packet

