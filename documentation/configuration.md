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
model_validation=true
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
**Description:** List of models to host.  

For **command arguments** and **environment variables**, use comma-separated key=value pairs:
```bash
--hosted-models "id=MODEL_ID,device=DEVICE,memory=GB,load_ends=BOOL"
```

**Example:**
```bash
--hosted-models "id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=false"
```

For **TOML configuration**, use the array of tables syntax:
```toml
[[hosted_models]]
id = "Qwen/Qwen3-1.7B"
device = "cpu"
max_memory = 4
load_ends = false
```

**Properties:**
| Key | Required | Type | Description |
|-----|----------|------|-------------|
| `id` | Yes | string | HuggingFace model ID or file path to model inside `/models` folder |
| `device` | Yes | string | PyTorch device type (e.g., `"cuda:0"`, `"cpu"`) |
| `max_memory` / `memory` | Yes | decimal | Maximum memory in GB to use for this model |
| `load_ends` | No | bool | Whether to load embedding/head layers (default: `false`) |

## Optional Properties

### `logging_level`
**Command Argument:** `--logging-level`  
**Environment Variable:** `LP_LOGGING_LEVEL`  
**Type:** `String`  
**Default:** `"INFO"`  
**Allowed Values:** `"INFO" | "DEBUG" | "WARNING" | "ERROR"`  
**Description:** Level of verbosity for the server to print to standard out. Sets the [internal logger's log level](https://docs.python.org/3/library/logging.html#logging-levels).  

### [`oai_port`](./oai.md)
**Command Argument:** `--openai-port`  
**Environment Variable:** `LP_OAI_PORT`  
**Type:** `Int`  
**Default:** `None`  
**Allowed Values:** Valid port number  
**Description:** Port for OpenAI-compatible server. No OpenAI server will be hosted if this field is left out.  

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
**Command Argument:** `--bootstrap-port`  
**Environment Variable:** `LP_BOOTSTRAP_PORT`  
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

### `job_port`
**Command Argument:** `--job-port`  
**Environment Variable:** `LP_JOB_PORT`  
**Type:** `Int`  
**Default:** `5050`  
**Allowed Values:** Valid port number  
**Description:** Port for job communication.
