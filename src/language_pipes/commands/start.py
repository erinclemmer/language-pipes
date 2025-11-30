import os
import re
import toml
from pathlib import Path
from unique_names_generator import get_random_name

from language_pipes.config import LpConfig
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes import LanguagePipes
from language_pipes.util.user_prompts import prompt_bool, prompt, prompt_choice, prompt_number_choice
from language_pipes.util import sanitize_file_name

def start_wizard(apply_overrides, version: str):
    """First-time setup wizard: handles network key, config, and starts server."""
    print("""         
==============================================================================
 | |                                              |  __ (_)                
 | |     __ _ _ __   __ _ _   _  __ _  __ _  ___  | |__) | _ __   ___  ___ 
 | |    / _` | '_ \ / _` | | | |/ _` |/ _` |/ _ \ |  ___/ | '_ \ / _ \/ __|
 | |___| (_| | | | | (_| | |_| | (_| | (_| |  __/ | |   | | |_) |  __/\__ \\
 |______\__,_|_| |_|\__, |\__,_|\__,_|\__, |\___| |_|   |_| .__/ \___||___/
                     __/ |             __/ |              | |              
                    |___/             |___/               |_|      
==============================================================================
""")

    default_base_path = Path(os.path.expanduser("~") ) / ".language-pipes"
    app_dir = default_base_path
    if not os.path.exists(default_base_path):
        app_dir = prompt("Where should we store application data?", required=True, default=str(default_base_path))
    
    if not os.path.exists(app_dir):
        Path(app_dir).mkdir(parents=True)
        print(f"Created directory: {app_dir}")

    config_dir = str(default_base_path / "configs")

    if not os.path.exists(config_dir):
        Path(config_dir).mkdir(parents=True)

    existing_configs = []
    if len(os.listdir(config_dir)) > 0:
        existing_configs = ["New Configuration"] + [f.replace(".toml", "") for f in os.listdir(config_dir)]

    existing_configs.reverse()

    load_config = "New Configuration"
    if len(existing_configs) > 1:
        load_config = prompt_number_choice("Load Configuration", existing_configs, default="New Configuration")
        if load_config is None:
            exit()
        load_config = load_config + ".toml"

    if load_config == "New Configuration.toml":
        raw_name = prompt("Name of new configuration", default=get_random_name())
        load_config = sanitize_file_name(raw_name)
        if not load_config or load_config == ".toml":
            load_config = "config.toml"
        interactive_init(str(Path(config_dir) / load_config))

    config_path = str(Path(config_dir) / load_config)

    print(f"Configuration: {load_config}")
    if not os.path.exists(config_path):
        print(f"Cannot find config at path {config_path}")
        exit()
    with open(config_path, 'r', encoding='utf-8') as f:
        print(f.read())

    use_config = prompt_bool("Use this configuration?", default=True)
    if not use_config:
        return start_wizard(apply_overrides, version)
    
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
            network_key = None
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
            "app_data_dir": app_dir,
            "router": {
                "node_id": data["node_id"],
                "port": data["peer_port"],
                "credential_dir": str(Path(app_dir) / "credentials"),
                "aes_key_file": data["network_key"],
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
        print("Exiting Language Pipes...")
