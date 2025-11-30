import os
import toml

from language_pipes.config import LpConfig
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes import LanguagePipes
from language_pipes.util.user_prompts import prompt_bool, prompt

def key_wizard(default_path):
    print("--- Step 1: Network Key ---\n")
    print("A network key is required for secure communication between nodes.")
    print("All nodes in your network must share the same key.\n")
    
    use_existing = prompt_bool("Use an existing key file?", default=False)
    if use_existing:
        selected_path = prompt("Existing key file", default=default_path, required=True)
        if os.path.exists(selected_path):
            print(f"✓ Network key found at '{selected_path}'")
            return selected_path
        else:
            while True:
                print(f"Error cannot find file at {selected_path}")
                again = prompt_bool("Try again?", default=True)
                if not again:
                    use_existing = False
                    break
                selected_path = prompt("Existing key file", default=default_path, required=True)

    if not use_existing:
        is_first = prompt_bool("Are you the first node in this network?", default=True)
        if is_first:
            create_key = prompt_bool("Use encrypted connection for network data?", default=False)
            if create_key:
                selected_path = prompt("Path for key generation", default=default_path, required=True)
                print(f"\nGenerating new network key...")
                key = save_new_aes_key(selected_path)
                print(f"✓ Network key generated: {key}")
                print(f"✓ Network key saved to '{selected_path}'")
                print("\n  IMPORTANT: Copy this key file to all other nodes that will")
                print("  join your network. Keep it secure - anyone with this key can join.\n")
                input("Press Enter to continue")
                return selected_path
            else:
                print("A network key is not needed, continuing...")
                return None
        else:
            use_key = prompt_bool("Does the network use encryption?", default=False)
            if use_key:
                print("\n  Enter the network key from your first node.")
                print("  (This is the hex string that was displayed when the key was generated)\n")
                key_value = prompt("Network key: ", required=True)
                
                with open(default_path, 'w', encoding='utf-8') as f:
                    f.write(key_value)
                
                print(f"✓ Network key saved to '{default_path}'")
                return default_path
            else:
                print("Continuing initialization...")
                return None
    return None

def start_wizard(config_path: str, key_path: str, apply_overrides, version: str):
    """First-time setup wizard: handles network key, config, and starts server."""
    print("\n" + "=" * 50)
    print("  Language Pipes - Quick Start")
    print("=" * 50)
    print("\nThis wizard will help you get started with Language Pipes.\n")

    # Step 1: Network Key
    selected_key_path = key_wizard(key_path)

    # Step 2: Configuration
    print("\n--- Step 2: Configuration ---\n")
    
    if os.path.exists(config_path):
        print(f"✓ Configuration found at '{config_path}'")
        if prompt_bool("Would you like to reconfigure?", default=False):
            interactive_init(config_path, selected_key_path)
    else:
        print(f"No configuration file found at '{config_path}'")
        print("Let's create one now.\n")
        interactive_init(config_path, selected_key_path)
        
        if not os.path.exists(config_path):
            print("\n✗ Configuration was not created. Exiting.")
            return

    # Step 3: Start Server
    print("\n--- Step 3: Start Server ---\n")
    
    print(f"Configuration: {config_path}")
    print(f"Network key:   {key_path}")
    print()
    
    if prompt_bool("Start the Language Pipes server now?", default=True):
        print("\nStarting server...\n")
        print("=" * 50)
        
        # Load config and start
        with open(config_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        
        # Create a minimal args-like object for apply_overrides
        class Args:
            logging_level = None
            openai_port = None
            node_id = None
            peer_port = None
            bootstrap_address = None
            bootstrap_port = None
            network_key = key_path
            model_validation = None
            ecdsa_verification = None
            job_port = None
            max_pipes = None
            hosted_models = None
        
        args = Args()
        data = apply_overrides(data, args)
        
        config = LpConfig.from_dict({
            "logging_level": data["logging_level"],
            "oai_port": data["oai_port"],
            "router": {
                "node_id": data["node_id"],
                "port": data["peer_port"],
                "aes_key_file": key_path,
                "bootstrap_nodes": [
                    {
                        "address": data["bootstrap_address"],
                        "port": data["bootstrap_port"]
                    }
                ] if data["bootstrap_address"] is not None else []
            },
            "processor": {
                "max_pipes": data["max_pipes"],
                "model_validation": data["model_validation"],
                "ecdsa_verification": data["ecdsa_verification"],
                "job_port": data["job_port"],
                "hosted_models": data["hosted_models"]
            }
        })

        return LanguagePipes(version, config)
    else:
        print("\nSetup complete! Start the server later with:")
        print(f"  language-pipes serve -c {config_path}")
