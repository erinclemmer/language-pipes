TIPS = {
    "network": {
        "configure": {
            "node_id": "A unique name that identifies this computer on the network.\nOther nodes will use this to route jobs to this machine.",
            "network_key": "An AES encryption key shared by all nodes. It encrypts\ncommunication and prevents unauthorized access.\nLeave empty for no encryption",
            "network_ip": "The IP address other nodes will use to connect to this node.",
            "peer_port": "The peer port is used for network coordination and discovery.\nOther nodes will connect to this port to join the network.",
            "bootstrap_nodes": "A list of nodes that this node can reach to connect to\nthe rest of the network"
        }
    }
}