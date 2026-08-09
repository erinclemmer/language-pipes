TIPS = {
    "network": {
        "configure": {
            "node_id": "Node ID: A unique name that identifies this computer on the network.\nOther nodes will use this to route jobs to this machine.",
            "network_key": "Network Key: An AES encryption key shared by all nodes. It encrypts\ncommunication and prevents unauthorized access.\nLeave empty for no encryption",
            "network_ip": "Network IP: The IP address other nodes will use to connect to this node.",
            "peer_port": "Peer Port: Used for network coordination and discovery.\nOther nodes will connect to this port to join the network.",
            "bootstrap_nodes": "Bootstrap Nodes: A list of nodes that this node can reach to connect to\nthe rest of the network",
            "whitelist_node_ids": "Whitelist Node IDs: A list of nodes that this node is allowed to communicate\nwith. Keep the list empty to allow all nodes that pass authentication\nto communicate."
        }
    },
    "layer_models": {
        "model_id": "Model ID: A HuggingFace model ID (e.g. Qwen/Qwen3-1.7B).\nThe model must be installed before it can be hosted.",
        "device": "Device: The PyTorch device to load layers onto.\nPress Enter to choose between cpu and any available cuda devices.",
        "max_memory": "Max Memory: The maximum amount of memory (in GB) to allocate for\nthis model's layers on the chosen device. Higher values\nallow more layers to be loaded on this node.",
        "data_type": "Data Type: The data type to load the model in.\nRequires the bitsandbytes package to change.",
        "bf16": "BF16: Loads model in 16 bit precision data type.",
        "int8": "int8: 8 bit integer data type used with bitsandbytes' LLM.int8() GPU kernels.\nLimited CPU support, requires AVX512F or AVX512BF16 CPU architecture\nto be efficient.",
        "int4": "int4: 4 bit integer data type used with bitsandbytes' QLORA GPU kernels.\nLimited CPU support, requires AVX512F or AVX512BF16 CPU architecture\nto be efficient."
    },
    "end_models": {
        "model_id": "Model ID: A HuggingFace model ID (e.g. Qwen/Qwen3-1.7B).\nThe model must be installed before it can be hosted.",
        "device": "Device: The PyTorch device used for both the local layers and the\nembedding/output head modules of this end model.\nPress Enter to choose between cpu and any available cuda devices.",
        "local_layers": "Local Layers: The number of transformer layers to run locally on this\ntrusted node alongside the embedding and output head. The remaining\nlayers run on layer models hosted by peers across the network."
    },
    "jobs_server": {
        "port": "Port: The network port the OpenAI-compatible server listens on.\nClients send inference requests to this port.",
        "max_node_jobs": "Max Node Jobs: The maximum number of jobs this node will process\nconcurrently per node on the network.",
        "max_api_jobs": "Max API Jobs: The maximum number of inference requests the server\nwill handle concurrently per API key.If no API keys are set\nthis is a global limit.",
        "api_keys": "API Keys: Keys that clients must provide to authenticate with the\nserver. Leave the list empty to allow unauthenticated access."
    }
}