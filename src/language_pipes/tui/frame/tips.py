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
    }
}