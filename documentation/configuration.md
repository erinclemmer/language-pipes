# Configuration

Below is an example configuration:
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

### Explanation
**logging_level:** Level of verbosity.  
[**oai_port:**](./oai.md) (int, optional) Port for openai compatable server, no OpenAI server will be hosted if this field is left out.  
  
**router.node_id:** (string) String identifier for the your server, must be unique on the network.  
**router.port:** (int) Port for the peer-to-peer network communication.  
**router.aes_key_file:** (string) RSA encryption key for the network.  
  
[**processor.https:**](./https.md) (boolean) Whether to communicate in https (true) or http (false) mode for slightly less latency at the cost of security.  
**processor.job_port:** (int) Port for job communication.  
**processor.hosted_models:** (list) List of models to host.  
**processor.hosted_models[].id:** (string )Huggingface ID or file path to model inside of "/models" folder.  
**processor.hosted_models[].device:** (string) Device type to host on, corresponds to pytorch device type e.g. "cuda:0", "cpu", etc.
**processor.hosted_models[].max_memory:** (decimal) (in GB) Maximum memory to use to host this model.  