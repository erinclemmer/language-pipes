# Configuration

```json
{
    "logging_level": "info",
    "oai_port": 8080,
    "router": {
        "node_id": "node-1",
        "port": 5000,
        "aes_key_file":"network.key",
        "bootstrap_nodes": [
            {
                "address": "192.0.0.10",
                "port": 5000
            }
        ]
    },
    "processor": {
        "https": true,
        "job_port": 5001,
        "hosted_models": [
            {
                "id": "cool-model",
                "device": "cuda",
                "max_memory": 5
            }
        ]
    }
}
```