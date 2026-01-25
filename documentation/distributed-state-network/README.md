# Distributed State Network

A Python framework for building distributed applications where nodes automatically share state without explicit data requests.

## Basis
Distributed state network is built as a helper to Language Pipes. It is meant to serve as a default technology that is based on pre-existing research. The basis of this research is as follows:

### The Maintenance of Duplicate Databases (Paul R. Johnson, Robert H. Thomas)
[Original RFC](https://www.rfc-editor.org/rfc/rfc677.html)  
Language Pipes requires a distributed database to function correctly. The ideas of direct mail updates were first established here and have been implemented in Distributed State Network.

### Epidemic algorithms in replicated databases (extended abstract)
[Paper](https://dl.acm.org/doi/pdf/10.1145/263661.263680)  
Distributed state network uses a very simple implementation of the gossip protocol described in this paper. Whenever the network is checking connection health every few seconds, the network randomly selects a node to send an update to. The paper outlines that this is the best complexity to benifit option to ensure that all nodes are eventually synced. Since this is meant to run on a local network the chance of packet loss is relatively low.

## Quick Start

### 1. Create Your First Node

The simplest DSN network is a single node:

```python
from distributed_state_network import DSNodeServer, DSNodeConfig

# Start a node
node = DSNodeServer.start(DSNodeConfig(
    node_id="my_first_node",
    port=8000,
    bootstrap_nodes=[]  # Empty for the first node
))

# Write some data
node.node.update_data("status", "online")
node.node.update_data("temperature", "72.5")
```

## How It Works

DSN creates a peer-to-peer network where each node maintains its own state database:

**Key concepts:**
- Each node owns its state and is the only one who can modify it
- State changes are automatically broadcast to all connected nodes
- Any node can read any other node's state instantly
- All communications can be encrypted with AES

## Example: Distributed Temperature Monitoring

Create a network of temperature sensors that share readings:

```python
# On each Raspberry Pi with a sensor:
sensor_node = DSNodeServer.start(DSNodeConfig(
    node_id=f"sensor_{location}",
    port=8000,
    bootstrap_nodes=[{"address": "coordinator.local", "port": 8000}]
))

# Continuously update temperature
while True:
    temp = read_temperature_sensor()
    sensor_node.node.update_data("temperature", str(temp))
    sensor_node.node.update_data("timestamp", str(time.time()))
    time.sleep(60)
```

On the monitoring station:
```python
for node_id in monitor.node.peers():
    if node_id.startswith("sensor_"):
        temp = monitor.node.read_data(node_id, "temperature")
        print(f"{node_id}: {temp}°F")
```
  
### Learn more
* [Configuration Class](./ds-node-config.md)
* [Server Class](./ds-node-server.md)
* [Protocol](./protocol.md)
