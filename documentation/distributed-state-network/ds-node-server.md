---
title: DSNodeServer
description: The HTTP server of a DSNode. The server handles the incoming requests of the Distributed State Network.
---

## DSNodeServer

`DSNodeServer` is the HTTP server of a DSNode. The server handles the incoming requests of the network.

```python
from distributed_state_network import DSNodeServer
```

### Class definition

```python
class DSNodeServer:
    config: DSNodeConfig
    network_ip: Optional[str]
    running: bool
    node: DSNode
    thread: Optional[threading.Thread]
    http_server: Optional[ThreadingHTTPServer]
```

### Constructor

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `DSNodeConfig` | The configuration of the node. |
| `create_alert` | `Callable[[str], None]` | A function that shows an alert message to the operator. |
| `disconnect_callback` | `Optional[Callable]` | A function that the node calls at a disconnect event. |
| `update_callback` | `Optional[Callable]` | A function that the node calls at a state update event. |
| `receive_callback` | `Optional[Callable]` | A function that the node calls when it receives data from a different node. |

**NOTE:** The constructor does not start the HTTP server. To make the instance and to start the server, use the static method `start()`.

### Static methods

#### `start() -> DSNodeServer`

This method makes a new `DSNodeServer` instance and starts the threaded HTTP server. If the configuration has bootstrap nodes, the method connects to the first bootstrap node that responds. If no bootstrap node responds, the method calls `create_alert`.

```python
server = DSNodeServer.start(config, alert)
```

**Parameters:**

The parameters are the same as the parameters of the constructor.

**Returns:**

- `DSNodeServer`: The server instance that runs.

**Example with a bootstrap node:**

```python
def alert(message: str) -> None:
    print(message)


# Bootstrap node (the first node in the network)
bootstrap_config = DSNodeConfig.from_dict({
    "node_id": "bootstrap",
    "port": 8000,
    "bootstrap_nodes": []
})
bootstrap = DSNodeServer.start(bootstrap_config, alert)

# Connector node (joins the existing network)
connector_config = DSNodeConfig.from_dict({
    "node_id": "connector",
    "port": 8001,
    "bootstrap_nodes": [{"address": "127.0.0.1", "port": 8000}]
})
connector = DSNodeServer.start(connector_config, alert)
```

#### `generate_key() -> str`

This method makes a new AES-128 key for the encryption of the network. The key has 32 hexadecimal characters (16 bytes).

**CAUTION:** Give the same key to all the nodes in the network. A node that has a different key cannot communicate with the network.

**Parameters:**

- None

**Example:**

```python
DSNodeServer.generate_key()
```

### Instance methods

#### `stop() -> None`

This method stops the server and releases the resources of the server.

**Example:**

```python
server.stop()
```

#### `update_data(key: str, value: str) -> None`

This method changes one key-value pair in the state of the node. Then the node sends the update to all its peers.

```python
server.update_data("status", "active")
```

**Parameters:**

- `key` (`str`): The key in the state to change.
- `value` (`str`): The new value of the key.

#### `read_data(node_id: str, key: str) -> Optional[str]`

This method reads one value from the state of a given node.

**Parameters:**

- `node_id` (`str`): The identifier of the node to read from.
- `key` (`str`): The key to read.

**Returns:**

- `Optional[str]`: The value of the key. The method returns `None` if the key does not exist.

#### `peers() -> List[str]`

This method gives the identifiers of all the connected peers.

**Parameters:**

- None

**Returns:**

- `List[str]`: The list of the node identifiers.

#### `send_to_node(node_id: str, data: bytes) -> None`

This method sends data to a different node.

```python
server.send_to_node('node-1', b'foo bar')
```

**Parameters:**

- `node_id` (`str`): The identifier of the node to send the data to.
- `data` (`bytes`): The data to send to the node.

**Returns:**

- None

#### `is_shut_down() -> bool`

This method gives the status of the server.

```python
server.is_shut_down()
```

**Parameters:**

- None

**Returns:**

- `bool`: The value is `True` if the server is shut down.

#### `node_id() -> str`

This method gives the identifier of this node.

```python
server.node_id()
```

**Parameters:**

- None

**Returns:**

- `str`: The identifier of this node.

#### `set_receive_cb(cb: Callable) -> None`

This method sets the receive callback. The node calls this function each time that it receives a data packet.

```python
def receive(node_id: str, data: bytes) -> None:
    pass


server.set_receive_cb(receive)
```

**Parameters:**

- `cb` (`Callable[[str, bytes], None]`): A function that receives the identifier of the sender and the bytes of the data packet.

**Returns:**

- None

#### `set_update_cb(cb: Callable) -> None`

This method sets the update callback. The node calls this function each time that a peer sends a state update.

```python
def update_cb() -> None:
    pass


server.set_update_cb(update_cb)
```

**Parameters:**

- `cb` (`Callable[[], None]`): A function that receives no arguments.

**Returns:**

- None

#### `set_disconnect_cb(cb: Callable) -> None`

This method sets the disconnect callback. The node calls this function each time that a peer disconnects from the network.

```python
def disconnect() -> None:
    pass


server.set_disconnect_cb(disconnect)
```

**Parameters:**

- `cb` (`Callable[[], None]`): A function that receives no arguments.

**Returns:**

- None

## Network protocol

The server uses HTTP. The protocol has these properties:

- **Encryption**: The nodes can encrypt all the packets with AES-128-CBC.
- **Authentication**: The nodes sign the packets with ECDSA.

For more information, refer to [Protocol](./protocol.md).

## Message types

The server handles five types of messages:

1. **HELLO (1)**: The node sends its information and its public key.
2. **PEERS (2)**: The node asks for the list of peers, or sends the list.
3. **UPDATE (3)**: The node sends a change of its state.
4. **PING (4)**: The node does a check of the health of the connection.
5. **DATA (5)**: The node sends a data packet.
