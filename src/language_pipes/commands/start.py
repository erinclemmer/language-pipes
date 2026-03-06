import os
import toml
import requests
from time import sleep
from pathlib import Path
from typing import Optional

from language_pipes.distributed_state_network import DSNodeServer, DSNodeConfig

from language_pipes.config import LpConfig, apply_env_overrides
from language_pipes.config import default_model_dir, default_config_dir
from language_pipes.commands.initialize import interactive_init
from language_pipes.commands.edit import edit_config
from language_pipes.commands.view import view_config
from language_pipes.lp import LanguagePipes
from language_pipes.util.user_prompts import prompt, prompt_number_choice, select_config, get_config_files, show_banner
from language_pipes.util import sanitize_file_name

def check_latest_version() -> Optional[str]:
    res = requests.get('https://raw.githubusercontent.com/erinclemmer/language-pipes/refs/heads/main/pyproject.toml')
    if res.status_code != 200:
        return None
    try:
        data = toml.loads(res.content.decode())
        return data['project']['version']
    except Exception as e:
        print(e)
        return None

def start_server(app_dir: str, config_path: str):
    show_banner("Starting server")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    data = apply_env_overrides(data, {"app_dir": app_dir})
    
    config = LpConfig.from_dict(data)

    try:
        router = DSNodeServer.start(DSNodeConfig.from_dict({
            "node_id": data.get("node_id"),
            "port": data.get("peer_port", 5000),
            "network_ip": data.get("network_ip", None),
            "credential_dir": str(Path(app_dir) / "credentials"),
            "aes_key": data.get("network_key", None),
            "whitelist_ips": data.get("whitelist_ips", []),
            "whitelist_node_ids": data.get("whitelist_node_ids", []),
            "bootstrap_nodes": [
                {
                    "address": data["bootstrap_address"],
                    "port": data["bootstrap_port"]
                }
            ] if data.get("bootstrap_address") is not None else []
        }))
        
        LanguagePipes(config, router)
        while True:
            sleep(60)
    except KeyboardInterrupt:
        exit()
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
    if config_path is not None:
        edit_config(config_path)

def start_wizard(version: str):
    print(f"""         
==============================================================================
  _                                                ____
 | |                                              |  __`(_)                
 | |     __ _ _ __   __ _ _   _  __ _  __ _  ___  | |__) | _ __   ___  ___ 
 | |    / _` | '_ \\ / _` | | | |/ _` |/ _` |/ _ \\ |  ___/ | '_ \\ / _ \\/ __|
 | |___| (_| | | | | (_| | |_| | (_| | (_| |  __/ | |   | | |_) |  __/\\__ \\
 |______\\__,_|_| |_|\\__, |\\__,_|\\__,_|\\__, |\\___| |_|   |_| .__/ \\___||___/
                     __/ |             __/ |              | |              
                    |___/             |___/               |_|      
Version: {version}
==============================================================================

""")

    app_dir = default_config_dir()
    model_dir = default_model_dir()
    tag = "- Made with <3 by Erin"
    latest_version = check_latest_version()
    if latest_version is not None and latest_version != version:
        tag = f"Version {latest_version} now available!\nInstall with: pip install --upgrade language-pipes"
    print(tag)
    print("")

    if not os.path.exists(app_dir):
        Path(app_dir).mkdir(parents=True)
        print(f"Created directory: {app_dir}")
    
    config_dir = str(Path(app_dir) / "configs")
    if not os.path.exists(config_dir):
        Path(config_dir).mkdir(parents=True)

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True)

    while True:
        main_menu_cmd = prompt_number_choice("Main Menu", [
            "View Config",
            "Load Config",
            "Create Config",
            "Edit Config",
            "Delete Config"
        ])
        if main_menu_cmd is None:
            exit()

        print("\n" + ("=" * 50) + "\n")

        match main_menu_cmd:
            case "Load Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                config_path = select_config(app_dir)
                if config_path is not None:
                    view_config(config_path)
                    start_server(app_dir, config_path)
            case "View Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                config_path = select_config(app_dir)
                if config_path is not None:
                    view_config(config_path)
            case "Create Config":
                try:
                    new_config(app_dir)
                except KeyboardInterrupt:
                    pass
            case "Edit Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                try:
                    modify_config(app_dir)
                except KeyboardInterrupt:
                    pass
            case "Delete Config":
                if len(get_config_files(config_dir)) == 0:
                    print("No configs found...\n\n")
                    continue
                delete_config(app_dir)

        print("\n" + ("=" * 50) + "\n")
