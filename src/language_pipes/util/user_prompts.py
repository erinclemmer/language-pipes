import os
from pathlib import Path

def prompt(message: str, default=None, required=False) -> str | None:
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


def prompt_int(message: str, default=None, required=False) -> int | None:
    """Prompt user for integer input."""
    while True:
        value = prompt(message, default=str(default) if default else None, required=required)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid number.")


def prompt_float(message: str, default=None, required=False) -> float | None:
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


def prompt_choice(message: str, choices: list, default=None) -> str | None:
    """Prompt user to select from choices."""
    choices_str = "/".join(choices)
    while True:
        value = prompt(f"{message} ({choices_str})", default=default)
        if value in choices:
            return value
        print(f"  Please choose from: {choices_str}")

def prompt_number_choice(message: str, choices: list, default=None, required=False) -> str:
    print(message)
    for i, choice  in enumerate(choices):
        print(f"[{i}] {choice}")
    if default is not None:
        default_idx = choices.index(default)
    else:
        default_idx = None
    while True:
        try:
            strSelect = prompt("Select number of choice", default=default_idx, required=required)
            if strSelect is None:
                continue
            selection = int(strSelect)
        except KeyboardInterrupt:
            exit()
        except:
            print("Invalid selection")
            continue
        if selection < 0 or selection >= len(choices):
            print("Invalid selection")
        else:
            return choices[selection]
    return None

def prompt_continue():
    prompt("Press Enter to continue...")

def get_config_files(config_dir: str):
    return [f.replace(".toml", "") for f in os.listdir(config_dir)]

def select_config(app_dir: str) -> str | None:
    config_dir = str(Path(app_dir) / "configs")
    existing_configs = get_config_files(config_dir)

    if len(existing_configs) > 0:
        load_config = prompt_number_choice("Select Configuration", existing_configs, required=True)
        if load_config is None:
            exit()
        load_config = load_config + ".toml"
    else:
        print("No configs found...")
        return None

    return str(Path(config_dir) / load_config)