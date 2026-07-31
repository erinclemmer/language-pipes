---
title: DSNodeConfig
description: The configuration object of a DSNode instance in the Distributed State Network.
---

## DSNodeConfig

`DSNodeConfig` is the configuration object of a DSNode instance.

```python
from distributed_state_network import DSNodeConfig
```

### Class definition

```python
@dataclass(frozen=False)
class DSNodeConfig:
    node_id: str
    logging_dir: Path
    credential_dir: Path
    port: int
    network_ip: Optional[str]
    aes_key: Optional[str]
    bootstrap_nodes: List[Endpoint]
    whitelist_node_ids: List[str]
```

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `node_id` | `str` | `""` | The identifier of the node. It must be different from the identifier of each other node. |
| `logging_dir` | `Path` | `logs` | The directory for the log files of the node. |
| `credential_dir` | `Path` | `credentials` | The directory for the ECDSA credentials of the node. |
| `port` | `int` | `0` | The TCP port on which the HTTP server of the node listens. |
| `network_ip` | `Optional[str]` | `None` | The IP address of the node on the network. |
| `aes_key` | `Optional[str]` | `None` | The AES-128 key for the encryption of the network. Give the key in hexadecimal characters (16 bytes, or 32 characters). |
| `bootstrap_nodes` | `List[Endpoint]` | `[]` | The nodes to which this node connects when it joins the network. |
| `whitelist_node_ids` | `List[str]` | `[]` | The identifiers of the permitted peers. If the list is empty, all the identifiers are permitted. |

**NOTE:** If you do not set `network_ip`, the node detects its own IP address. If the node cannot detect the address, the bootstrap node uses the address of the incoming request.

**NOTE:** The `whitelist_ips` attribute is no longer available. Use `whitelist_node_ids` in its place. ECDSA authenticates a node ID, but an IP address is not a stable identity of a peer. The node ignores a `whitelist_ips` key in an existing configuration.

### Methods

#### `from_dict(data: Dict) -> DSNodeConfig`

This static method makes a `DSNodeConfig` instance from a dictionary. If the dictionary does not have a key, the method uses the default value from the table of attributes.

**Parameters:**

- `data` (`Dict`): The dictionary that contains the configuration parameters.

**Returns:**

- `DSNodeConfig`: The configuration instance.

**Example:**

```python
config_dict = {
    "node_id": "node1",
    "port": 8000,
    "bootstrap_nodes": [
        {"address": "127.0.0.1", "port": 8001}
    ]
}
config = DSNodeConfig.from_dict(config_dict)
```

**Example of a bootstrap node (the first node in the network):**

```python
config_dict = {
    "node_id": "bootstrap",
    "port": 8000,
    "bootstrap_nodes": []  # Empty for the first node
}
config = DSNodeConfig.from_dict(config_dict)
```
