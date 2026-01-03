import os
import socket
import toml
from unique_names_generator import get_random_name

from language_pipes.util.aes import generate_aes_key
from language_pipes.util.user_prompts import prompt, prompt_bool, prompt_choice, prompt_float, prompt_int, prompt_number_choice, prompt_continue

def get_default_node_id() -> str:
    return socket.gethostname()

def edit_config(config_path: str):
    """Edit an existing configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    while True:
        print("\n" + "=" * 50)
        print("  Edit Configuration")
        print("=" * 50)
        
        # Display current configuration
        print("\nCurrent Configuration:")
        print("-" * 30)
        print(toml.dumps(config))
        print("-" * 30)
        
        # Build list of editable properties
        editable_props = []
        prop_keys = []
        
        # Simple properties
        simple_props = [
            ("node_id", "Node ID"),
            ("oai_port", "OpenAI API Port"),
            ("peer_port", "Peer Port"),
            ("job_port", "Job Port"),
            ("network_ip", "Network IP"),
            ("bootstrap_address", "Bootstrap Address"),
            ("bootstrap_port", "Bootstrap Port"),
            ("network_key", "Network Key"),
            ("logging_level", "Logging Level"),
            ("max_pipes", "Max Pipes"),
            ("model_validation", "Model Validation"),
            ("ecdsa_verification", "ECDSA Verification"),
            ("print_times", "Print Times"),
            ("print_job_data", "Print Job Data"),
            ("prefill_chunk_size", "Prefill Chunk Size"),
        ]
        
        for key, label in simple_props:
            current_val = config.get(key, "Not set")
            editable_props.append(f"{label}: {current_val}")
            prop_keys.append(key)
        
        # Hosted models as special entry
        model_count = len(config.get("hosted_models", []))
        editable_props.append(f"Hosted Models ({model_count} configured)")
        prop_keys.append("hosted_models")
        
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
        
        print()
        
        if selected_key == "node_id":
            config["node_id"] = prompt(
                "Node ID",
                default=config.get("node_id", get_default_node_id()),
                required=True
            )
        
        elif selected_key == "oai_port":
            current = config.get("oai_port")
            if prompt_bool("Enable OpenAI API?", default=current is not None):
                config["oai_port"] = prompt_int(
                    "API Port",
                    default=current or 8000,
                    required=True
                )
            else:
                config.pop("oai_port", None)
        
        elif selected_key == "peer_port":
            config["peer_port"] = prompt_int(
                "Peer Port",
                default=config.get("peer_port", 5000),
                required=True
            )
        
        elif selected_key == "job_port":
            config["job_port"] = prompt_int(
                "Job Port",
                default=config.get("job_port", 5050),
                required=True
            )
        
        elif selected_key == "network_ip":
            config["network_ip"] = prompt(
                "Network IP",
                default=config.get("network_ip", "127.0.0.1"),
                required=True
            )
        
        elif selected_key == "bootstrap_address":
            current = config.get("bootstrap_address")
            if prompt_bool("Connect to bootstrap node?", default=current is not None):
                config["bootstrap_address"] = prompt(
                    "Bootstrap Address",
                    default=current,
                    required=True
                )
            else:
                config.pop("bootstrap_address", None)
        
        elif selected_key == "bootstrap_port":
            config["bootstrap_port"] = prompt_int(
                "Bootstrap Port",
                default=config.get("bootstrap_port", 5000),
                required=True
            )
        
        elif selected_key == "network_key":
            current = config.get("network_key")
            if prompt_bool("Enable network encryption?", default=current is not None):
                new_key = prompt(
                    "Network Key",
                    default=current or "Generate new key"
                )
                if new_key == "Generate new key":
                    key = generate_aes_key().hex()
                    config["network_key"] = key
                    print(f"Generated new key: {key}")
                else:
                    config["network_key"] = new_key
            else:
                config.pop("network_key", None)
        
        elif selected_key == "logging_level":
            config["logging_level"] = prompt_choice(
                "Logging Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                default=config.get("logging_level", "INFO")
            )
        
        elif selected_key == "max_pipes":
            config["max_pipes"] = prompt_int(
                "Max Pipes",
                default=config.get("max_pipes", 1),
                required=True
            )
        
        elif selected_key == "model_validation":
            config["model_validation"] = prompt_bool(
                "Enable model hash validation?",
                default=config.get("model_validation", False)
            )
        
        elif selected_key == "ecdsa_verification":
            config["ecdsa_verification"] = prompt_bool(
                "Enable ECDSA signing?",
                default=config.get("ecdsa_verification", False)
            )
        
        elif selected_key == "print_times":
            config["print_times"] = prompt_bool(
                "Print timing info?",
                default=config.get("print_times", False)
            )
        
        elif selected_key == "print_job_data":
            config["print_job_data"] = prompt_bool(
                "Print job data?",
                default=config.get("print_job_data", False)
            )
        
        elif selected_key == "prefill_chunk_size":
            config["prefill_chunk_size"] = prompt_int(
                "Prefill Chunk Size (tokens, prompts longer than this are chunked)",
                default=config.get("prefill_chunk_size", 128),
                required=True
            )
        
        elif selected_key == "hosted_models":
            config["hosted_models"] = edit_hosted_models(config.get("hosted_models", []))


def edit_hosted_models(models: list) -> list:
    """Edit the hosted models list."""
    while True:
        print("\n--- Hosted Models ---\n")
        
        options = []
        for i, model in enumerate(models):
            model_str = f"Model #{i+1}: {model.get('id', 'Unknown')} ({model.get('device', 'cpu')}, {model.get('max_memory', 4)}GB)"
            options.append(model_str)
        
        options.append("Add new model")
        options.append("Done editing models")
        
        selection = prompt_number_choice("Select model to edit", options, required=True)
        
        if selection == "Done editing models":
            return models
        
        if selection == "Add new model":
            new_model = edit_single_model({
                "id": "Qwen/Qwen3-1.7B",
                "device": "cpu",
                "max_memory": 4,
                "load_ends": True
            })
            if new_model:
                models.append(new_model)
            continue
        
        # Edit existing model
        model_idx = options.index(selection)
        
        action = prompt_number_choice("Action", ["Edit", "Delete", "Cancel"], required=True)
        
        if action == "Delete":
            if len(models) == 1:
                print("Cannot delete the last model. At least one model is required.")
                prompt_continue()
            else:
                models.pop(model_idx)
                print("Model deleted")
        elif action == "Edit":
            edited = edit_single_model(models[model_idx])
            if edited:
                models[model_idx] = edited


def edit_single_model(model: dict) -> dict | None:
    """Edit a single model configuration."""
    print("\n--- Edit Model ---\n")
    
    props = [
        ("id", "Model ID"),
        ("device", "Device"),
        ("max_memory", "Max Memory (GB)"),
        ("load_ends", "Load Ends"),
        ("Done", "Done editing"),
        ("Cancel", "Cancel changes"),
    ]
    
    edited_model = model.copy()
    
    while True:
        print(f"\nCurrent: {edited_model.get('id')} on {edited_model.get('device')}, {edited_model.get('max_memory')}GB, ends={edited_model.get('load_ends')}")
        
        options = [f"{label}: {edited_model.get(key, 'N/A')}" if key not in ("Done", "Cancel") else label for key, label in props]
        
        selection = prompt_number_choice("Select property", options, required=True)
        selected_idx = options.index(selection)
        selected_key = props[selected_idx][0]
        
        if selected_key == "Done":
            return edited_model
        
        if selected_key == "Cancel":
            return None
        
        if selected_key == "id":
            edited_model["id"] = prompt(
                "Model ID (HuggingFace ID)",
                default=edited_model.get("id"),
                required=True
            )
        
        elif selected_key == "device":
            edited_model["device"] = prompt(
                "Device (cpu, cuda:0, etc.)",
                default=edited_model.get("device", "cpu"),
                required=True
            )
        
        elif selected_key == "max_memory":
            edited_model["max_memory"] = prompt_float(
                "Max Memory (GB)",
                default=edited_model.get("max_memory", 4),
                required=True
            )
        
        elif selected_key == "load_ends":
            edited_model["load_ends"] = prompt_bool(
                "Load embedding/output layers?",
                required=True
            )
