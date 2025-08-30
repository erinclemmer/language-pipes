# Configuration

Below is an example toml configuration:
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

#### `logging_level`
**Type:** `String`  
**Required:** No  
**Default:** `"INFO"`  
**Allowed Values:** `"INFO" | "DEBUG" | "WARNING" | "ERROR"`  
**Description:** Level of verbosity for the server to print to standard out.  

#### [`oai_port`](./oai.md)
**Type:** `Int`  
**Required:** No  
**Default:** `None`  
**Allowed Values:** Valid port number  
**Description:** Port for openai compatable server, no OpenAI server will be hosted if this field is left out.  

##### `node_id`
**Type:** String  
**Required:** Yes    
**Description:**  String identifier for your server, must be unique on the network.  

#### `peer_port`
**Type:** `Int`  
**Required:** No  
**Default:** `5000`  
**Description:** Port for the peer-to-peer network communication.  

#### `bootstrap_address`
**Type:** `String`  
**Required:** No    
**Description:** Address to reach out to when connecting to the network.  

#### `bootstrap_port`
**Type:** `Int`  
**Required:** No    
**Description:** port for `bootstrap_address`.  

#### `network_key`
**Type:** `String`  
**Required:** No  
**Default:** `"network.key"`  
**Allowed Values:** Valid path  
**Description:** RSA encryption key for the network.  

#### [`https`](./https.md)
**Type:** `Bool`  
**Required:** No  
**Default:** False  
**Allowed Values:** `true | false`  
**Description:** Whether to communicate in https (true) or http (false) mode for slightly less latency at the cost of security.  


#### `job_port`
**Type:** `Int`  
**Required:** No  
**Default:** `5050`  
**Allowed Values:** Valid port number  
**Description:** Port for job communication.  

#### `hosted_models`
**Type:** `Array`  
**Required:** Yes    
**Description:** List of models to host.  
**processor.hosted_models[].id:** (string) Huggingface ID or file path to model inside of "/models" folder.  
**processor.hosted_models[].device:** (string) Device type to host on, corresponds to pytorch device type e.g. "cuda:0", "cpu", etc.  
**processor.hosted_models[].max_memory:** (decimal) (in GB) Maximum memory to use to host this model.  