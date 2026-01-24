## Usage Examples

### Basic Setup

```python
from distributed_state_network import DSNodeServer, DSNodeConfig, Endpoint

# Create configuration
config = DSNodeConfig(
    node_id="node1",
    port=8000,
    bootstrap_nodes=[]  # Empty for first node
)

# Start server (UDP socket will be created automatically)
server = DSNodeServer.start(config)

# Update state
server.node.update_data("status", "online")
server.node.update_data("version", "1.0.0")

# Read own state
my_status = server.node.read_data("node1", "status")

# Get connected peers
peers = server.node.peers()
print(f"Connected peers: {peers}")

# Shutdown
server.stop()
```

### Joining Existing Network

```python
# Node 2 configuration with bootstrap
config2 = DSNodeConfig(
    node_id="node2",
    port=8001,
    bootstrap_nodes=[
        Endpoint("127.0.0.1", 8000)  # Node 1's endpoint
    ]
)

server2 = DSNodeServer.start(config2)

# Read state from peer
peer_status = server2.node.read_data("node1", "status")
```

### With Custom Socket

```python
import socket

# Create a custom UDP socket
custom_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
custom_socket.bind(("0.0.0.0", 8003))

# You can configure socket options before passing it to the server
custom_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

config = DSNodeConfig(
    node_id="node3",
    port=8003,  # Should match the bound port
    bootstrap_nodes=[Endpoint("127.0.0.1", 8000)]
)

# Pass the custom socket when starting the server
server = DSNodeServer.start(config, sock=custom_socket)
```

### With Disconnect Callback

```python
def handle_disconnect():
    print("A peer has disconnected!")
    # Handle reconnection logic

config = DSNodeConfig(
    node_id="node4",
    port=8004,
    bootstrap_nodes=[Endpoint("127.0.0.1", 8000)]
)

server = DSNodeServer.start(config, disconnect_callback=handle_disconnect)
```

### With Update Callback

```python
def handle_update():
    print("A peer has updated their state!")
    # React to state changes

config = DSNodeConfig(
    node_id="node5",
    port=8005,
    bootstrap_nodes=[Endpoint("127.0.0.1", 8000)]
)

server = DSNodeServer.start(config, update_callback=handle_update)
```

### Complete Example with All Options

```python
import socket
from distributed_state_network import DSNodeServer, DSNodeConfig, Endpoint

def on_peer_disconnect():
    print("Peer disconnected from network")

def on_state_update():
    print("Peer state was updated")

# Create custom socket (optional)
custom_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
custom_sock.bind(("0.0.0.0", 8006))

config = DSNodeConfig(
    node_id="node6",
    port=8006,
    bootstrap_nodes=[
        Endpoint("127.0.0.1", 8000),
        Endpoint("127.0.0.1", 8001)  # Multiple bootstrap options
    ]
)

# Start with all options
server = DSNodeServer.start(
    config,
    sock=custom_sock,  # Optional custom socket
    disconnect_callback=on_peer_disconnect,
    update_callback=on_state_update
)

# Use the server
server.node.update_data("role", "worker")
peers = server.node.peers()

# When done
server.stop()
```

### Network Protocol Notes

- All communication uses **UDP** instead of HTTP/HTTPS
- Packets are encrypted with AES using the shared key
- Messages are signed with ECDSA for authentication
- Maximum UDP packet size is 65507 bytes
- Request timeout is 2 seconds with up to 3 retry attempts
- Network health checks occur every 3 seconds
