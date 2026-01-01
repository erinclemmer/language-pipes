import os
import re
import toml
from pathlib import Path
from unique_names_generator import get_random_name

from language_pipes.config import LpConfig
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes.commands.edit import edit_config
from language_pipes.commands.view import view_config
from language_pipes import LanguagePipes
from language_pipes.util.user_prompts import prompt_bool, prompt, prompt_choice, prompt_number_choice, select_config, get_config_files
from language_pipes.util import sanitize_file_name

def start_server(apply_overrides, app_dir: str, config_path: str, version: str):
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
        network_ip = None
        app_dir = None
        bootstrap_address = None
        bootstrap_port = None
        network_key = None
        model_validation = None
        ecdsa_verification = None
        print_times = None
        print_job_data = None
        job_port = None
        max_pipes = None
        hosted_models = None
    
    args = Args()
    data = apply_overrides(data, args)
    
    config = LpConfig.from_dict({
        "logging_level": data["logging_level"],
        "oai_port": data["oai_port"],
        "app_dir": app_dir,
        "router": {
            "node_id": data["node_id"],
            "port": data["peer_port"],
            "network_ip": data["network_ip"],
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
            "print_times": data["print_times"],
            "print_job_data": data["print_job_data"],
            "hosted_models": data["hosted_models"]
        }
    })

    try:
        return LanguagePipes(version, config)
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(e)

def new_config(app_dir: str):
    raw_name = prompt("Name of new configuration", required=True)
    if raw_name is None:
        return
    file_name = sanitize_file_name(raw_name)
    if not file_name:
        print("Invalid file name")
        return
    
    file_path = str(Path(app_dir) / "configs" / file_name)

    interactive_init(file_path)

def delete_config(app_dir: str):
    config_path = select_config(app_dir)
    if config_path is not None:
        os.remove(config_path)
        print("Configuration deleted")

def modify_config(app_dir: str):
    config_path = select_config(app_dir)
    if config_path:
        edit_config(config_path)

def start_wizard(apply_overrides, version: str):
    print(f"""         
==============================================================================

 | |                                              |  __ (_)                
 | |     __ _ _ __   __ _ _   _  __ _  __ _  ___  | |__) | _ __   ___  ___ 
 | |    / _` | '_ \ / _` | | | |/ _` |/ _` |/ _ \ |  ___/ | '_ \ / _ \/ __|
 | |___| (_| | | | | (_| | |_| | (_| | (_| |  __/ | |   | | |_) |  __/\__ \\
 |______\__,_|_| |_|\__, |\__,_|\__,_|\__, |\___| |_|   |_| .__/ \___||___/
                     __/ |             __/ |              | |              
                    |___/             |___/               |_|      
Version: {version}
==============================================================================

- Made with <3 by Erin
""")

    app_dir = str(Path(os.path.expanduser("~") ) / ".config" / "language-pipes")
    
    if not os.path.exists(app_dir):
        Path(app_dir).mkdir(parents=True)
        print(f"Created directory: {app_dir}")
    
    config_dir = str(Path(app_dir) / "configs")
    if not os.path.exists(config_dir):
        Path(config_dir).mkdir(parents=True)

    models_dir = str(Path(app_dir) / "models")
    if not os.path.exists(models_dir):
        Path(models_dir).mkdir(parents=True)

    while True: # TODO Add model list        
        main_menu_cmd = prompt_number_choice("Main Menu", [
            "View Config",
            "Load Config",
            "Create Config",
            "Edit Config",
            "Delete Config"
        ])

        print('\n==============================================================================\n')

        match main_menu_cmd:
            case "Load Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                config_path = select_config(app_dir)
                if config_path is not None:
                    view_config(config_path)
                    return start_server(apply_overrides, app_dir, config_path, version)
            case "View Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                config_path = select_config(app_dir)
                if config_path is not None:
                    view_config(config_path)
            case "Create Config":
                new_config(app_dir)
            case "Edit Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                modify_config(app_dir)
            case "Delete Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                delete_config(app_dir)

        print('\n==============================================================================\n')
