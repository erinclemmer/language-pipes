## DSNodeServer

UDP server wrapper for DSNode that handles incoming network requests.

```python
from distributed_state_network import DSNodeServer
```

### Class Definition
```python
class DSNodeServer:
    config: DSNodeConfig
    node: DSNode
    socket: socket.socket
    running: bool
    thread: threading.Thread
```

### Constructor

**Parameters:**
- `config` (`DSNodeConfig`): Node configuration
- `sock` (`Optional[socket.socket]`): Custom UDP socket to use (optional, creates new socket if not provided)
- `disconnect_callback` (`Optional[Callable]`): Callback for disconnect events
- `update_callback` (`Optional[Callable]`): Callback for state update events

**Example:**
```python
# Create with default socket
server = DSNodeServer(config)

# Or with custom socket
custom_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
custom_socket.bind(("0.0.0.0", 8000))
server = DSNodeServer(config, sock=custom_socket)
```

### Static Methods

### `start(config: DSNodeConfig, sock: Optional[socket.socket] = None, disconnect_callback: Optional[Callable] = None, update_callback: Optional[Callable] = None) -> DSNodeServer`
Creates and starts a new DSNodeServer instance with UDP socket.

```python
server = DSNodeServer.start(config)
```

**Parameters:**
- `config` (`DSNodeConfig`): Node configuration
- `sock` (`Optional[socket.socket]`): Custom UDP socket to use (optional)
- `disconnect_callback` (`Optional[Callable]`): Callback for disconnect events (no parameters)
- `update_callback` (`Optional[Callable]`): Callback for state update events (no parameters)

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

### `serve_forever() -> None`
Main UDP server loop. Continuously listens for incoming packets and routes them to appropriate handlers. This method is typically called automatically by `start()` in a daemon thread.

**Note:** This method blocks until the server is stopped.

## Network Protocol

The server uses UDP sockets with the following characteristics:

- **Port**: Configurable via DSNodeConfig
- **Encryption**: All packets encrypted with AES
- **Authentication**: ECDSA signatures for message verification
- **Max Packet Size**: 65507 bytes (UDP limit)
- **Timeout**: 2 seconds with automatic retry (up to 3 attempts)

## Message Types

The server handles four types of messages:

1. **HELLO (1)**: Node introduction and public key exchange
2. **PEERS (2)**: Request/response for peer list
3. **UPDATE (3)**: State synchronization
4. **PING (4)**: Connection health check

## Automatic IP Detection

When a client connects to a bootstrap server:

1. Server extracts client's IP from UDP packet source address
2. Server includes detected IP in HELLO response via `detected_address` field
3. Client updates its own address book with the detected IP
4. Correct IP propagates to other nodes through peer discovery

This eliminates the need for manual IP configuration and works correctly behind NAT.

## Thread Safety

The server spawns separate threads for:
- Main UDP receive loop (`serve_forever`)
- Packet handling (one thread per packet)
- Node network tick (periodic health checks)

All shared state is protected with appropriate locks.

## Example: Multi-Node Setup

```python
from distributed_state_network import DSNodeServer, DSNodeConfig, Endpoint

# Generate shared AES key (only once)
key = DSNodeServer.generate_key()

# Start bootstrap node
bootstrap = DSNodeServer.start(DSNodeConfig(
    node_id="bootstrap",
    port=8000,
    aes_key=key,
    bootstrap_nodes=[]
))

# Start additional nodes
node1 = DSNodeServer.start(DSNodeConfig(
    node_id="node1",
    port=8001,
    aes_key=key,
    bootstrap_nodes=[Endpoint("192.168.1.100", 8000)]
))

node2 = DSNodeServer.start(DSNodeConfig(
    node_id="node2",
    port=8002,
    aes_key=key,
    bootstrap_nodes=[Endpoint("192.168.1.100", 8000)]
))

# Nodes automatically discover each other through bootstrap
```

## Troubleshooting

### Server won't start
- Check if port is already in use
- Verify AES key file exists and is readable
- Ensure firewall allows UDP traffic on specified port

### Nodes can't connect
- Verify all nodes use the same AES key
- Check bootstrap node is running and reachable
- Verify network connectivity (ping bootstrap address)
- Check firewall rules for UDP ports

### Connection drops
- Increase timeout if on slow networks
- Check network stability
- Verify nodes aren't being stopped unexpectedly
