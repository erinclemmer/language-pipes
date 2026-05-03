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
        "device": "Device: The PyTorch device to load layers onto.\nExamples: cpu, cuda:0, cuda:1, etc.",
        "max_memory": "Max Memory: The maximum amount of memory (in GB) to allocate for\nthis model's layers on the chosen device. Higher values\nallow more layers to be loaded on this node."
    }
}