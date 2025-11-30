def prompt(message: str, default=None, required=False) -> str:
    """Prompt user for input with optional default value."""
    if default is not None:
        display = f"{message} [{default}]: "
    else:
        display = f"{message}: "
    
    while True:
        value = input(display).strip()
        if value == "":
            if default is not None:
                return default
            if required:
                print("  This field is required.")
                continue
            return None
        return value


def prompt_int(message: str, default=None, required=False) -> int:
    """Prompt user for integer input."""
    while True:
        value = prompt(message, default=str(default) if default else None, required=required)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid number.")


def prompt_float(message: str, default=None, required=False) -> float:
    """Prompt user for float input."""
    while True:
        value = prompt(message, default=str(default) if default else None, required=required)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            print("  Please enter a valid number.")


def prompt_bool(message: str, default=False) -> bool:
    """Prompt user for yes/no input."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{message} [{default_str}]: ").strip().lower()
        if value == "":
            return default
        if value in ("y", "yes", "true", "1"):
            return True
        if value in ("n", "no", "false", "0"):
            return False
        print("  Please enter 'y' or 'n'.")


def prompt_choice(message: str, choices: list, default=None) -> str:
    """Prompt user to select from choices."""
    choices_str = "/".join(choices)
    while True:
        value = prompt(f"{message} ({choices_str})", default=default)
        if value in choices:
            return value
        print(f"  Please choose from: {choices_str}")


def get_default_node_id() -> str:
    """Generate a default node ID based on hostname."""
    try:
        hostname = socket.gethostname()
        return f"node-{hostname}"
    except:
        return "node-1"


def interactive_init(output_path: str):
    """Interactively create a configuration file."""
    print("\n" + "=" * 50)
    print("  Language Pipes Configuration Setup")
    print("=" * 50)
    print("\nThis wizard will help you create a config.toml file.")
    print("Press Enter to accept the default value shown in [brackets].\n")

    config = {}

    # === Required Settings ===
    print("--- Required Settings ---\n")

    config["node_id"] = prompt(
        "Node ID (unique identifier for this node)",
        default=get_default_node_id(),
        required=True
    )

    # === Model Configuration ===
    print("\n--- Model Configuration ---\n")
    print("You need to specify at least one model to host.")
    print("Models are downloaded from HuggingFace by ID.\n")

    hosted_models = []
    while True:
        print(f"  Model #{len(hosted_models) + 1}:")
        
        model_id = prompt(
            "    HuggingFace model ID (e.g., Qwen/Qwen3-1.7B)",
            default="Qwen/Qwen3-1.7B" if len(hosted_models) == 0 else None,
            required=len(hosted_models) == 0
        )
        
        if model_id is None:
            break
        
        device = prompt(
            "    Device (cpu, cuda:0, cuda:1, etc.)",
            default="cpu",
            required=True
        )
        
        max_memory = prompt_float(
            "    Max memory in GB",
            default=4,
            required=True
        )
        
        load_ends = prompt_bool(
            "    Load embedding/head layers (needed for first/last node)",
            default=False
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

    if prompt_bool("Enable OpenAI-compatible API server?", default=True):
        config["oai_port"] = prompt_int(
            "  API server port",
            default=6000,
            required=True
        )

    # === Network Configuration ===
    print("\n--- Network Configuration ---\n")

    is_first_node = prompt_bool(
        "Is this the first node in the network?",
        default=True
    )

    if not is_first_node:
        config["bootstrap_address"] = prompt(
            "  Bootstrap node IP address",
            required=True
        )
        config["bootstrap_port"] = prompt_int(
            "  Bootstrap node port",
            default=5000,
            required=True
        )

    config["peer_port"] = prompt_int(
        "Peer-to-peer network port",
        default=5000
    )

    config["job_port"] = prompt_int(
        "Job communication port",
        default=5050
    )

    config["network_key"] = prompt(
        "Network key file path",
        default="network.key"
    )

    # === Advanced Options ===
    print("\n--- Advanced Options ---\n")

    if prompt_bool("Configure advanced options?", default=False):
        config["logging_level"] = prompt_choice(
            "  Logging level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO"
        )
        
        config["max_pipes"] = prompt_int(
            "  Maximum pipes to participate in",
            default=1
        )
        
        config["model_validation"] = prompt_bool(
            "  Validate model weight hashes?",
            default=False
        )
        
        config["ecdsa_verification"] = prompt_bool(
            "  Enable ECDSA packet signing?",
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

    if prompt_bool(f"Write configuration to '{output_path}'?", default=True):
        # Check if file exists
        if os.path.exists(output_path):
            if not prompt_bool(f"  '{output_path}' already exists. Overwrite?", default=False):
                print("Aborted.")
                return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            toml.dump(clean_config, f)
        
        print(f"\n✓ Configuration saved to '{output_path}'")
        
        # Helpful next steps
        print("\nNext steps:")
        if is_first_node:
            print(f"  1. Generate network key:  language-pipes keygen {config.get('network_key', 'network.key')}")
            print(f"  2. Start the server:      language-pipes serve -c {output_path}")
        else:
            print(f"  1. Copy network key from the first node")
            print(f"  2. Start the server:      language-pipes serve -c {output_path}")
    else:
        print("Configuration not saved.")
