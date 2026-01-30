import socket
import toml
from typing import List

from language_pipes.commands.view import view_config
from language_pipes.util.aes import generate_aes_key
from language_pipes.util.user_prompts import prompt, prompt_bool, prompt_choice, prompt_float, prompt_int, prompt_number_choice, show_banner, prompt_model_id

def edit_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    while True:
        show_banner("Edit Configuration")        
        view_config(config_path)
        
        editable_props = []
        prop_keys = []
        
        simple_props = [
            ("node_id", "Node ID"),
            ("oai_port", "OpenAI API Port"),
            ("peer_port", "Peer Port"),
            ("network_ip", "Network IP"),
            ("bootstrap_address", "Bootstrap Address"),
            ("bootstrap_port", "Bootstrap Port"),
            ("network_key", "Network Key"),
            ("logging_level", "Logging Level"),
            ("max_pipes", "Max Pipes"),
            ("model_validation", "Model Validation"),
            ("prefill_chunk_size", "Prefill Chunk Size"),
        ]
        
        model_count = len(config.get("hosted_models", []))
        editable_props.append(f"Hosted Models ({model_count} configured)")
        prop_keys.append("hosted_models")
        
        for key, label in simple_props:
            current_val = config.get(key, "Not set")
            editable_props.append(f"{label}: {current_val}")
            prop_keys.append(key)
        
        editable_props.append("Save and Exit")
        prop_keys.append("__save__")
        
        editable_props.append("Exit without Saving")
        prop_keys.append("__cancel__")
        
        selection = prompt_number_choice("\nSelect property to edit", editable_props, required=True)
        selected_key = prop_keys[editable_props.index(selection)]
        
        if selected_key == "__save__":
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            print("\n✓ Configuration saved")
            return
        
        if selected_key == "__cancel__":
            print("\nChanges discarded")
            return
        
        if selected_key == "node_id":
            config["node_id"] = prompt(
                "Node ID",
                default=config.get("node_id", socket.gethostname()),
                required=True
            )
        
        elif selected_key == "oai_port":
            current = config.get(selected_key)
            if prompt_bool("Enable OpenAI API?", default=current is not None):
                config[selected_key] = prompt_int(
                    "API Port",
                    default=current or 8000,
                    required=True
                )
            else:
                config.pop(selected_key, None)
        
        elif selected_key == "peer_port":
            config[selected_key] = prompt_int(
                "Peer Port",
                default=config.get(selected_key, 5000),
                required=True
            )
        
        elif selected_key == "network_ip":
            config[selected_key] = prompt(
                "Network IP",
                default=config.get(selected_key, None),
                required=True
            )
        
        elif selected_key == "bootstrap_address":
            current = config.get(selected_key)
            if prompt_bool("Connect to bootstrap node?", default=current is not None):
                config[selected_key] = prompt(
                    "Bootstrap Address",
                    default=current,
                    required=True
                )
            else:
                config.pop(selected_key, None)
        
        elif selected_key == "bootstrap_port":
            config[selected_key] = prompt_int(
                "Bootstrap Port",
                default=config.get(selected_key, 5000),
                required=True
            )
        
        elif selected_key == "network_key":
            current = config.get(selected_key)
            if prompt_bool("Enable network encryption?", default=current is not None):
                new_key = prompt(
                    "Network Key",
                    default=current or "Generate new key"
                )
                if new_key == "Generate new key":
                    key = generate_aes_key().hex()
                    config[selected_key] = key
                    print(f"Generated new key: {key}")
                else:
                    config[selected_key] = new_key
            else:
                config.pop(selected_key, None)
        
        elif selected_key == "logging_level":
            config[selected_key] = prompt_choice(
                "Logging Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                default=config.get(selected_key, "INFO")
            )
        
        elif selected_key == "max_pipes":
            config[selected_key] = prompt_int(
                "Max Pipes",
                default=config.get(selected_key, 1),
                required=True
            )
        
        elif selected_key == "model_validation":
            config[selected_key] = prompt_bool(
                "Enable model hash validation?",
                default=config.get(selected_key, False)
            )
        
        elif selected_key == "prefill_chunk_size":
            config[selected_key] = prompt_int(
                "Prefill Chunk Size",
                default=config.get(selected_key, 6),
                required=True
            )
        
        elif selected_key == "hosted_models":
            try:
                config["hosted_models"] = edit_hosted_models(config.get("hosted_models", []))
            except KeyboardInterrupt:
                continue

def select_hosted_model(models: List[dict]) -> int | None:
    options = []
    for i, model in enumerate(models):
        model_str = f"Model #{i+1}: {model.get('id', 'Unknown')} ({model.get('device', 'cpu')}, {model.get('max_memory', 4)}GB)"
        options.append(model_str)
    options.append("Cancel")
    model_selection = prompt_number_choice("Select Model", options)
    if model_selection == "Cancel":
        return None
    return options.index(model_selection)

def edit_hosted_models(models: List) -> List[dict]:
    while True:
        print("\n--- Hosted Models ---\n")
        
        options = [
            "Add model",
            "Edit model",
            "Delete Model",
            "Done editing models"
        ]
        
        selection = prompt_number_choice("Select model to edit", options, required=True)
        
        if selection == "Add model":
            new_model = {
                "id": prompt_model_id("Model ID", True),
                "device": prompt("Device", "cpu", required=True),
                "max_memory": prompt("Max Memory (GB)"),
                "load_ends": prompt_bool("Load Ends", required=True)
            }
            if new_model:
                models.append(new_model)
            continue
        
        if selection == "Edit model":
            model_idx = select_hosted_model(models)
            if model_idx is None:
                continue
            try:
                edited = edit_single_model(models[model_idx])
                if edited:
                    models[model_idx] = edited
            except KeyboardInterrupt:
                continue

        if selection == "Delete model":
            model_idx = select_hosted_model(models)
            if model_idx is None:
                continue
            models.pop(model_idx)

        if selection == "Done editing models":
            return models

def edit_single_model(model: dict) -> dict | None:
    print("\n--- Edit Model ---\n")
    
    props = [
        ("id", "Model ID"),
        ("device", "Device"),
        ("max_memory", "Max Memory (GB)"),
        ("load_ends", "Load Ends")
    ]
    
    edited_model = model.copy()
    
    while True:
        options = [f"{label}: {edited_model.get(key, 'N/A')}" for key, label in props]
        options.append("Cancel")
        options.append("Done")
        
        selection = prompt_number_choice("Select property", options, required=True)
        if selection == "Cancel":
            return None
        
        if selection == "Done":
            return edited_model

        selected_idx = options.index(selection)
        selected_key = props[selected_idx][0]
        
        if selected_key == "id":
            edited_model[selected_key] = prompt_model_id("Model ID", required=True)
        
        elif selected_key == "device":
            edited_model[selected_key] = prompt(
                "Device (cpu, cuda:0, etc.)",
                default=edited_model.get(selected_key, "cpu"),
                required=True
            )
        
        elif selected_key == "max_memory":
            edited_model[selected_key] = prompt_float(
                "Max Memory (GB)",
                required=True
            )
        
        elif selected_key == "load_ends":
            edited_model["load_ends"] = prompt_bool(
                "Load embedding/output layers?",
                required=True
            )
