# distributed-state-network

A peer-to-peer network of nodes that share replicated state over HTTP, with
per-message AES encryption and ECDSA-signed packets. Each node gossips its
state to peers; the network self-organizes from a set of bootstrap nodes.

This is the networking layer used by
[language-pipes](https://github.com/erinclemmer/language-pipes), but has no
dependency on it and can be used on its own.

## Install

```bash
pip install distributed-state-network
```

## Usage

```python
from distributed_state_network import DSNodeServer, DSNodeConfig

key = DSNodeServer.generate_key()
server = DSNodeServer.start(DSNodeConfig(
    node_id="node-a",
    network_ip="127.0.0.1",
    port=8000,
    aes_key=key,
))
```

See `src/distributed_state_network/` for the packet types, key management, and
node handler implementation.

## License

MIT
