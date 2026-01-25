## DSNodeConfig

Configuration object for initializing a DSNode instance.

```python
from distributed_state_network import DSNodeConfig
```

### Class Definition
```python
@dataclass(frozen=True)
class DSNodeConfig:
    node_id: str
    port: int
    credential_dir: str
    aes_key: Optional[str]
    network_ip: Optional[str]
    bootstrap_nodes: List[Endpoint]
```

### Attributes
- **node_id** (`str`): Unique identifier for the node
- **port** (`int`): Port number for the node to listen on (UDP)
- **credential_dir**: (`str`) directory to store ECDSA credentials (default: "[current_directory]/credentials")
- **network_ip** (`str`): Network ip address (only required if other nodes will connect to you)
- **aes_key** (`str`): Hexidecimal encoded AES key for network encryption
- **bootstrap_nodes** (`List[Endpoint]`): List of initial nodes to connect to when joining the network

**Note:** If `network_ip` is not supplied, the node's public IP address is automatically detected by the bootstrap server during the initial handshake.

### Methods

### `from_dict(data: Dict) -> DSNodeConfig`
Creates a DSNodeConfig instance from a dictionary.

**Parameters:**
- `data` (`Dict`): Dictionary containing configuration parameters

**Returns:**
- `DSNodeConfig`: Configuration instance

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

**Example for bootstrap node (first node in network):**
```python
config_dict = {
    "node_id": "bootstrap",
    "port": 8000,
    "bootstrap_nodes": []  # Empty for first node
}
config = DSNodeConfig.from_dict(config_dict)
