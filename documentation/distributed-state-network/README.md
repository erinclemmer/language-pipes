---
title: Distributed State Network
description: A Python framework for distributed applications. The nodes share their state automatically, thus an application does not send a request for the data.
---

Distributed State Network (DSN) is a Python framework for distributed applications. Each node sends its state to the other nodes automatically. An application does not send a request for the data.

## Basis

DSN is a helper for Language Pipes. It is the default technology for shared state in Language Pipes. Two research documents give the basis of the design.

### The Maintenance of Duplicate Databases (Paul R. Johnson, Robert H. Thomas)

[Original RFC](https://www.rfc-editor.org/rfc/rfc677.html)

Language Pipes needs a distributed database. This document was the first to describe direct mail updates. DSN uses this method.

### Epidemic algorithms in replicated databases (extended abstract)

[Paper](https://dl.acm.org/doi/pdf/10.1145/263661.263680)

DSN uses a simple form of the gossip protocol from this paper. Each node does a check of the health of its connections every 3 seconds. At each check, the node selects one peer at random and sends an update to that peer. The paper shows that this method gives the best ratio of complexity to benefit. Thus all the nodes become synchronized after a short time. DSN operates on a local network, where the loss of packets is not frequent.

## Quick start

### 1. Create the first node

The most simple DSN network has one node. To create this node, do these steps:

1. Import `DSNodeServer` and `DSNodeConfig`.
2. Set `bootstrap_nodes` to an empty list. The first node in a network has no bootstrap node.
3. Start the node with `DSNodeServer.start()`.
4. Write the data to the state of the node with `update_data()`.

```python
from distributed_state_network import DSNodeServer, DSNodeConfig


def alert(message: str) -> None:
    print(message)


# Start a node
node = DSNodeServer.start(
    DSNodeConfig.from_dict({
        "node_id": "my_first_node",
        "port": 8000,
        "bootstrap_nodes": []  # Empty for the first node
    }),
    alert
)

# Write some data
node.update_data("status", "online")
node.update_data("temperature", "72.5")
```

## How it works

DSN makes a peer-to-peer network. Each node keeps its own state database.

Key concepts:

- Each node owns its state. Only that node can change its state.
- The node sends each change of its state to all the connected nodes automatically.
- A node can read the state of each other node immediately.
- The nodes can encrypt all communication with AES-128-CBC. Each message has a new random IV.

## Example: a distributed temperature monitor

This example shows a network of temperature sensors. Each sensor writes its temperature to its own state. The monitor station reads the temperature from each sensor.

Do these steps on each Raspberry Pi that has a sensor:

```python
sensor_node = DSNodeServer.start(
    DSNodeConfig.from_dict({
        "node_id": f"sensor_{location}",
        "port": 8000,
        "bootstrap_nodes": [{"address": "coordinator.local", "port": 8000}]
    }),
    alert
)

# Update the temperature every 60 seconds
while True:
    temp = read_temperature_sensor()
    sensor_node.update_data("temperature", str(temp))
    sensor_node.update_data("timestamp", str(time.time()))
    time.sleep(60)
```

Do this step on the monitor station:

```python
for node_id in monitor.peers():
    if node_id.startswith("sensor_"):
        temp = monitor.read_data(node_id, "temperature")
        print(f"{node_id}: {temp}°F")
```

### More information

* [Configuration class](./ds-node-config.md)
* [Server class](./ds-node-server.md)
* [Protocol](./protocol.md)
