import os
import socket
import toml

from language_pipes.util.aes import generate_aes_key
from language_pipes.commands.view import view_config
from language_pipes.util.user_prompts import prompt, prompt_bool, prompt_choice, prompt_float, prompt_int, prompt_model_id, show_banner

def interactive_init(output_path: str):
    show_banner("Configuration Setup")

    config = { }

    print("--- Required Settings ---\n")

    print("The node ID is a unique name that identifies this computer on the")
    print("Language Pipes network. Other nodes will use this to route jobs to this machine.\n")
    config["node_id"] = prompt(
        "    Node ID",
        default=socket.gethostname(),
        required=True
    )

    print("\n--- Layer Models ---\n")
    print("Here you can specify which models to host")
    print("Models are automatically downloaded from HuggingFace by their ID")
    print("if they do not exist in the model folder already.\n")

    layer_models = []
    if prompt_bool("Add layer models?", required=True):
        try:
            while True:
                print(f"    Model #{len(layer_models) + 1}:")
                
                print("    Select a locally available model or enter a HuggingFace model ID")
                model_id = prompt_model_id(
                    "    Model ID",
                    required=True
                )
                if model_id is None:
                    raise KeyboardInterrupt()
                
                print("\n    Select the compute device for this model. Use 'cpu' for CPU-only,")
                print("    or 'cuda:0', 'cuda:1', etc. for specific GPUs.")
                device = prompt(
                    "    Device",
                    default="cpu",
                    required=True
                )
                
                print("\n    Specify the maximum RAM/VRAM (in GB) this node should use for the model.")
                print("    The model layers will be loaded until this limit is reached.")
                max_memory = prompt_float(
                    "    Max memory (GB)",
                    required=True
                )
                
                layer_models.append({
                    "id": model_id,
                    "device": device,
                    "max_memory": max_memory
                })
                
                print()
                if not prompt_bool("Add another model?", required=True):
                    break
        except KeyboardInterrupt:
            pass

    config["layer_models"] = layer_models

    print("\n--- End Models ---\n")
    print("Here you can specify which end models to load")
    print("To run a model in the network you need an end model on the machine that hosts the Open AI compatible server")
    end_models = []
    if prompt_bool("Add end model?", required=True):
        try:
            while True:
                print(f"    End model #{len(end_models)}")

                model_id = prompt_model_id("    Model ID", required=True)
                if model_id is None:
                    break
                end_models.append(model_id)
                print(f"Added {model_id} to end models")

                if not prompt_bool("Add another model?", required=True):
                    break

        except KeyboardInterrupt:
            pass

    config["end_models"] = end_models

    # === API Server ===
    print("\n--- API Server ---\n")

    print("Language Pipes can expose an OpenAI-compatible HTTP API, allowing you to")
    print("use standard OpenAI client libraries (Python, JavaScript, curl, etc.)")
    print("to interact with your distributed model.\n")
    if prompt_bool("Enable OpenAI-compatible API server?", default=True):
        print("\n  Choose a port for the API server. Clients will connect to")
        print("  http://<ip-address>:<port>/v1/chat/completions")
        config["oai_port"] = prompt_int(
            "  API port",
            default=8000,
            required=True
        )

    # === Network Configuration ===
    print("\n--- Network Configuration ---\n")

    print("Language Pipes uses a peer-to-peer network to coordinate between nodes.")
    print("The first node starts fresh; additional nodes connect to an existing node.\n")
    is_first_node = prompt_bool("Is this the first node in the network?", required=True)

    if not is_first_node:
        print("\n  Enter the IP address of an existing node on the network.")
        print("  This node will connect to it to join the distributed network.")
        config["bootstrap_address"] = prompt(
            "  Bootstrap node IP",
            required=True
        )
        print("\n  Enter the peer port of the bootstrap node (default is 5000).")
        config["bootstrap_port"] = prompt_int(
            "  Bootstrap node port",
            default=5000,
            required=True
        )

    print("\nThe peer port is used for network coordination and discovery.")
    print("Other nodes will connect to this port to join the network.")
    config["peer_port"] = prompt_int(
        "Peer port",
        default=5000
    )

    if is_first_node:
        print("\nThe network IP is the address other nodes will use to connect to this node.")
        config["network_ip"] = prompt(
            "Network IP",
            required=True
        )

    print("\nThe network key is an AES encryption key shared by all nodes.")
    print("It encrypts communication and prevents unauthorized access.")
    print("There will be no encryption between nodes if the default value is selected.")
    print("")
    encrypt_traffic = prompt_bool("Encrypt network traffic", default=False)
    if encrypt_traffic:
        config["network_key"] = prompt(
            "Network key",
            default="Generate new key"
        )

        if config["network_key"] == "Generate new key":
            key = generate_aes_key().hex()
            config["network_key"] = key
            print(f"Generated new key: {key}")
            print("Note: Save this key somewhere and supply it to other nodes on the network")

    # === Advanced Options ===
    print("\n--- Advanced Options ---\n")

    print("Advanced options include logging verbosity, security features, and limits.")
    if prompt_bool("Configure advanced options?", default=False):
        print("\n  Controls how much information is printed to the console.")
        print("  DEBUG shows everything, ERROR shows only critical issues.")
        config["logging_level"] = prompt_choice(
            "  Logging level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO"
        )
        
        print("\n  Limits how many model 'pipes' (distributed model instances)")
        print("  this node will participate in simultaneously.")
        config["max_pipes"] = prompt_int(
            "  Max pipes",
            default=1
        )
        
        print("\n  When enabled, nodes verify that model weight hashes match")
        print("  to ensure all nodes are running the exact same model.")
        config["model_validation"] = prompt_bool(
            "  Validate model hashes?", 
            required=True
        )
        
    # === Write Config File ===
    print("\n" + "=" * 50)
    print("  Configuration Summary")
    print("=" * 50 + "\n")

    # Build clean config (remove None values)
    clean_config = {k: v for k, v in config.items() if v is not None and k != "layer_models"}
    clean_config["layer_models"] = config["layer_models"]

    # Check if file exists
    if os.path.exists(output_path):
        if not prompt_bool(f"  '{output_path}' already exists. Overwrite?", default=False):
            print("Aborted.")
            return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        toml.dump(clean_config, f)

    view_config(output_path)
    
    print("\n✓ Configuration saved")
