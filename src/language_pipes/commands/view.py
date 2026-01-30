import toml

def view_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    print("\n" + "=" * 60)
    print("  Configuration Details")
    print(f"  Path: {config_path}")
    print("=" * 60)
    
    # Property descriptions
    descriptions = {
        "node_id": "Unique identifier for this node on the network",
        "oai_port": "Port for OpenAI-compatible API server",
        "peer_port": "Port for peer-to-peer network coordination",
        "network_ip": "IP address other nodes use to connect to this node",
        "bootstrap_address": "IP of existing node to join the network",
        "bootstrap_port": "Port of the bootstrap node",
        "network_key": "AES encryption key for secure communication",
        "logging_level": "Verbosity level (DEBUG/INFO/WARNING/ERROR)",
        "max_pipes": "Max concurrent distributed model instances",
        "model_validation": "Verify model weight hashes match across nodes",
    }
    
    # Display simple properties
    print("\n--- Node Settings ---")
    _print_property("Node ID", config.get("node_id"), descriptions["node_id"])
    
    print("\n--- Network Settings ---")
    _print_property("Network IP", config.get("network_ip"), descriptions["network_ip"])
    _print_property("Peer Port", config.get("peer_port"), descriptions["peer_port"])
    
    bootstrap = config.get("bootstrap_address")
    if bootstrap:
        _print_property("Bootstrap Address", bootstrap, descriptions["bootstrap_address"])
        _print_property("Bootstrap Port", config.get("bootstrap_port"), descriptions["bootstrap_port"])
    else:
        print("  Bootstrap Node:     Not configured (this is a standalone/first node)")
    
    network_key = config.get("network_key")
    if network_key:
        # Show truncated key for security
        display_key = network_key[:8] + "..." + network_key[-8:] if len(network_key) > 20 else network_key
        _print_property("Network Key", display_key, descriptions["network_key"])
    else:
        print("  Network Encryption: Disabled")
    
    print("\n--- API Settings ---")
    oai_port = config.get("oai_port")
    if oai_port:
        _print_property("OpenAI API Port", oai_port, descriptions["oai_port"])
    else:
        print("  OpenAI API:         Disabled")
    
    print("\n--- Performance & Security ---")
    _print_property("Logging Level", config.get("logging_level", "INFO"), descriptions["logging_level"])
    _print_property("Max Pipes", config.get("max_pipes", 1), descriptions["max_pipes"])
    _print_property("Model Validation", "Enabled" if config.get("model_validation") else "Disabled", descriptions["model_validation"])
    
    # Display hosted models
    models = config.get("hosted_models", [])
    print(f"\n--- Hosted Models ({len(models)}) ---")
    for i, model in enumerate(models):
        print(f"\n  Model #{i+1}:")
        print(f"    ID:          {model.get('id', 'Unknown')}")
        # print(f"                 (HuggingFace model identifier)")
        print(f"    Device:      {model.get('device', 'cpu')}")
        # print(f"                 (Compute device: cpu, cuda:0, cuda:1, etc.)")
        print(f"    Max Memory:  {model.get('max_memory', 4)} GB")
        # print(f"                 (Maximum RAM/VRAM to use for model layers)")
        print(f"    Load Ends:   {'Yes' if model.get('load_ends') else 'No'}")
        # print(f"                 (Load embedding layer and output head)")
    
    print("\n" + "=" * 60)


def _print_property(label: str, value, description: str):
    """Helper to print a property with its description."""
    print(f"  {label + ':':<18} {value}")
