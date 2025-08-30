# Configuration

There are several options to configure a server and type can take precedence over each other. The precedence is defined below:

`command arguments > environment variables > toml configuration > system defaults`

Below is an example Toml configuration:
```toml
logging_level="INFO"
oai_port=6000
node_id="node-1" # Required
peer_port=5000
bootstrap_address="192.168.0.1"
bootstrap_port=5000
network_key="network.key"
https=true
job_port=5050

[[hosted_models]] # Required
id="meta-llama/Llama-3.2-1B-Instruct"
device="cpu"
max_memory=5
```

### `logging_level`
**Command Argument:** `--logging-level`  
**Environment Variable:** `LP_LOGGING_LEVEL`  
**Type:** `String`  
**Required:** No  
**Default:** `"INFO"`  
**Allowed Values:** `"INFO" | "DEBUG" | "WARNING" | "ERROR"`  
**Description:** Level of verbosity for the server to print to standard out.  

### [`oai_port`](./oai.md)
**Command Argument:** `--oai-port`  
**Environment Variable:** `LP_OAI_PORT`  
**Type:** `Int`  
**Required:** No  
**Default:** `None`  
**Allowed Values:** Valid port number  
**Description:** Port for openai compatable server, no OpenAI server will be hosted if this field is left out.  

### `node_id`
**Command Argument:** `--node-id`  
**Environment Variable:** `LP_NODE_ID`  
**Type:** String  
**Required:** Yes    
**Description:**  String identifier for your server, must be unique on the network.  

### `peer_port`
**Command Argument:** `--peer-port`  
**Environment Variable:** `LP_PEER_PORT`  
**Type:** `Int`  
**Required:** No  
**Default:** `5000`  
**Description:** Port for the peer-to-peer network communication.  

### `bootstrap_address`
**Command Argument:** `--bootstrap-address`  
**Environment Variable:** `LP_BOOTSTRAP_ADDRESS`  
**Type:** `String`  
**Required:** No    
**Description:** Address to reach out to when connecting to the network.  

### `bootstrap_port`
**Command Argument:** `--peer-port`  
**Environment Variable:** `LP_PEER_PORT`  
**Type:** `Int`  
**Required:** No    
**Description:** port for `bootstrap_address`.  

### `network_key`
**Command Argument:** `--network-key`  
**Environment Variable:** `LP_NETWORK_KEY`  
**Type:** `String`  
**Required:** No  
**Default:** `"network.key"`  
**Allowed Values:** Valid path  
**Description:** RSA encryption key for the network.  

### [`https`](./https.md)
**Command Argument:** `--https`  
**Environment Variable:** `LP_HTTPS`  
**Type:** `Bool`  
**Required:** No  
**Default:** False  
**Allowed Values:** `true | false`  
**Description:** Whether to communicate in https (true) or http (false) mode for slightly less latency at the cost of security.  


### `job_port`
**Command Argument:** `--job-port`  
**Environment Variable:** `LP_JOB_PORT`  
**Type:** `Int`  
**Required:** No  
**Default:** `5050`  
**Allowed Values:** Valid port number  
**Description:** Port for job communication.  

### `hosted_models`
**Command Argument:** `--hosted-models`  
**Environment Variable:** `LP_HOSTED_MODELS`  
**Type:** `Array`  
**Required:** Yes    
**Description:** List of models to host. For command arguments and environment variables it must be in this format: `[model-id]:[device]:[max_memory]`  
**processor.hosted_models[].id:** (string) Huggingface ID or file path to model inside of "/models" folder.  
**processor.hosted_models[].device:** (string) Device type to host on, corresponds to pytorch device type e.g. "cuda:0", "cpu", etc.  
**processor.hosted_models[].max_memory:** (decimal) (in GB) Maximum memory to use to host this model.  