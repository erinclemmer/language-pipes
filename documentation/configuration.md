# Configuration

There are several options to configure a server and the order of precedence is defined below:

`command arguments > environment variables > toml configuration > system defaults`

An example Toml configuration:
```toml
node_id="node-1" # Required
logging_level="INFO"
oai_port=6000
peer_port=5000
bootstrap_address="192.168.0.1"
bootstrap_port=5000
network_key="network.key"
ecdsa_verification=false
max_pipes=1
https=true
model_validation=true
network_ip="192.168.0.2"
job_port=5050

[[hosted_models]] # Required
id="meta-llama/Llama-3.2-1B-Instruct"
device="cpu"
max_memory=5
```

## Required Properties

### `node_id`
**Command Argument:** `--node-id`  
**Environment Variable:** `LP_NODE_ID`  
**Type:** String  
**Description:**  String identifier for your server, must be unique on the network.  

### `hosted_models`
**Command Argument:** `--hosted-models`  
**Environment Variable:** `LP_HOSTED_MODELS`  
**Type:** `Array`  
**Description:** List of models to host. For command arguments and environment variables it must be in this format: `[model-id]::[device]::[max_memory]`  
**processor.hosted_models[].id:** (string) Huggingface ID or file path to model inside of "/models" folder.  
**processor.hosted_models[].device:** (string) Device type to host on, corresponds to pytorch device type e.g. "cuda:0", "cpu", etc.  
**processor.hosted_models[].max_memory:** (decimal) (in GB) Maximum memory to use to host this model.  

## Optional Properties

### `logging_level`
**Command Argument:** `--logging-level`  
**Environment Variable:** `LP_LOGGING_LEVEL`  
**Type:** `String`  
**Default:** `"INFO"`  
**Allowed Values:** `"INFO" | "DEBUG" | "WARNING" | "ERROR"`  
**Description:** Level of verbosity for the server to print to standard out. Sets the [internal logger's log level](https://docs.python.org/3/library/logging.html#logging-levels).  

### [`oai_port`](./oai.md)
**Command Argument:** `--oai-port`  
**Environment Variable:** `LP_OAI_PORT`  
**Type:** `Int`  
**Default:** `None`  
**Allowed Values:** Valid port number  
**Description:** Port for openai compatible server, no OpenAI server will be hosted if this field is left out.  

### `peer_port`
**Command Argument:** `--peer-port`  
**Environment Variable:** `LP_PEER_PORT`  
**Type:** `Int`  
**Default:** `5000`  
**Description:** Port for the peer-to-peer network communication.  
Refer to the [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) package for more information.  

### `bootstrap_address`
**Command Argument:** `--bootstrap-address`  
**Environment Variable:** `LP_BOOTSTRAP_ADDRESS`  
**Type:** `String`  
**Description:** Address to reach out to when connecting to the network.  
Refer to the [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) package for more information.  

### `bootstrap_port`
**Command Argument:** `--peer-port`  
**Environment Variable:** `LP_PEER_PORT`  
**Type:** `Int`  
**Default:** 5000  
**Description:** port for `bootstrap_address`.  
Refer to the [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) package for more information.  

### `network_key`
**Command Argument:** `--network-key`  
**Environment Variable:** `LP_NETWORK_KEY`  
**Type:** `String`  
**Default:** `"network.key"`  
**Allowed Values:** Valid path  
**Description:** RSA encryption key for the network.  
Refer to the [Distributed State Network](https://github.com/erinclemmer/distributed_state_network) package for more information.  

### `https`
**Command Argument:** `--https`  
**Environment Variable:** `LP_HTTPS`  
**Type:** `Bool`  
**Default:** False    
**Description:** Whether to communicate in https (true) or http (false) mode for slightly less latency at the cost of security.  

### `model_validation`
**Command Argument:** `--model-validation`  
**Environment Variable:** `LP_MODEL_VALIDATION`  
**Type:** `Bool`  
**Default:** False    
**Description:** If set, it checks the weight hashes of other models on the network against the computed hashes of the local weights to determine if they are the same model.

### `ecdsa_verification`
**Command Argument:** `--ecdsa-verification`  
**Environment Variable:** `LP_ECDSA_VERIFICATION`  
**Type:** `Bool`  
**Default:** False    
**Description:** If set, uses the ecdsa algorithm to sign job packets so that the receiver will only accept job packets from pipes that it is a part of. 

### `max_pipes`
**Command Argument:** `--max-pipes`  
**Environment Variable:** `LP_MAX_PIPES`  
**Type:** `Int`      
**Description:** The maximum number of pipes to load models for.


### `network_ip`
**Command Argument:** `--network-ip`
**Environment Variable:** `LP_NETWORK_IP`
**Type:** `String`
**Description:** Network IP to create the node's HTTPS certificate. Should match your current local IP address.

### `job_port`
**Command Argument:** `--job-port`  
**Environment Variable:** `LP_JOB_PORT`  
**Type:** `Int`  
**Default:** `5050`  
**Allowed Values:** Valid port number  
**Description:** Port for job communication.  

