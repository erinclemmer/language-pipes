import os
import socket
import toml
from unique_names_generator import get_random_name

from language_pipes.util.aes import generate_aes_key
from language_pipes.util.user_prompts import prompt, prompt_bool, prompt_choice, prompt_float, prompt_int

def get_default_node_id() -> str:
    return socket.gethostname()

def interactive_init(output_path: str):
    """Interactively create a configuration file."""
    print("\n" + "=" * 50)
    print("  Configuration Setup")
    print("=" * 50)
    print("\nThis wizard will help you create a new configuration.")

    config = {}

    # === Required Settings ===
    print("--- Required Settings ---\n")

    print("The node ID is a unique name that identifies this computer on the")
    print("Language Pipes network. Other nodes will use this to route jobs.\n")
    config["node_id"] = prompt(
        "Node ID",
        default=get_default_node_id(),
        required=True
    )

    # === Model Configuration ===
    print("\n--- Model Configuration ---\n")
    print("Language Pipes hosts parts of large language models distributed across")
    print("multiple machines. You need to specify at least one model to host.")
    print("Models are automatically downloaded from HuggingFace by their ID.\n")

    hosted_models = []
    while True:
        print(f"  Model #{len(hosted_models) + 1}:")
        
        print("    Enter the HuggingFace model ID. This is the identifier used on")
        print("    huggingface.co (e.g., 'Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.2-1B-Instruct').")
        model_id = prompt(
            "    Model ID",
            default="Qwen/Qwen3-1.7B" if len(hosted_models) == 0 else None,
            required=len(hosted_models) == 0
        )
        
        if model_id is None:
            break
        
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
            default=4,
            required=True
        )
        
        print("\n    The 'ends' of a model are the embedding layer (input) and the output head.")
        print("    At least one node in the network needs these loaded to process requests.")
        print("    If this is the only node or the first/last in a chain, enable this.")
        load_ends = prompt_bool(
            "    Load embedding/output layers",
            default=True
        )
        
        hosted_models.append({
            "id": model_id,
            "device": device,
            "max_memory": max_memory,
            "load_ends": load_ends
        })
        
        print()
        if not prompt_bool("Add another model?", default=False):
            break

    config["hosted_models"] = hosted_models

    # === API Server ===
    print("\n--- API Server ---\n")

    print("Language Pipes can expose an OpenAI-compatible HTTP API, allowing you to")
    print("use standard OpenAI client libraries (Python, JavaScript, curl, etc.)")
    print("to interact with your distributed model.\n")
    if prompt_bool("Enable OpenAI-compatible API server?", default=True):
        print("\n  Choose a port for the API server. Clients will connect to")
        print("  http://<this-machine>:<port>/v1/chat/completions")
        config["oai_port"] = prompt_int(
            "  API port",
            default=8000,
            required=True
        )

    # === Network Configuration ===
    print("\n--- Network Configuration ---\n")

    print("Language Pipes uses a peer-to-peer network to coordinate between nodes.")
    print("The first node starts fresh; additional nodes connect to an existing node.\n")
    is_first_node = prompt_bool(
        "Is this the first node in the network?",
        default=True
    )

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
    else:
        config["network_ip"] = "127.0.0.1"

    print("\nThe job port is used for transferring computation data between nodes")
    print("during model inference (hidden states, embeddings, etc.).")
    config["job_port"] = prompt_int(
        "Job port",
        default=5050
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
    else:
        config["network_key"] = None

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
            default=False
        )
        
        print("\n  ECDSA verification signs each job packet cryptographically,")
        print("  ensuring jobs only come from authorized nodes in the pipe.")
        config["ecdsa_verification"] = prompt_bool(
            "  Enable ECDSA signing?",
            default=False
        )

    # === Write Config File ===
    print("\n" + "=" * 50)
    print("  Configuration Summary")
    print("=" * 50 + "\n")

    # Build clean config (remove None values)
    clean_config = {k: v for k, v in config.items() if v is not None and k != "hosted_models"}
    clean_config["hosted_models"] = config["hosted_models"]

    # Generate TOML preview
    preview = toml.dumps(clean_config)
    print(preview)

    # Check if file exists
    if os.path.exists(output_path):
        if not prompt_bool(f"  '{output_path}' already exists. Overwrite?", default=False):
            print("Aborted.")
            return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        toml.dump(clean_config, f)
    
    print(f"\n✓ Configuration saved")
