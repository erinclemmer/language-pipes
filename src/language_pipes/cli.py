import argparse
from importlib import resources
import os
from pathlib import Path
from typing import Optional
from language_pipes.config_args import ConfigurationArgs
from language_pipes.util.aes import generate_aes_key
from language_pipes.util.config import get_app_dir
from language_pipes.util.logging import setup_logging

VERSION = (
    resources.files("language_pipes")
    .joinpath("VERSION")
    .read_text(encoding="utf-8")
    .strip()
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Language Pipes",
        description="Peer-to-peer distributed inference for open-source language models",
    )

    parser.add_argument("-v", "--version", action="version", version=VERSION)
    parser.add_argument("-c", "--config", help="Load configuration from TOML file")
    parser.add_argument("--start", help="Start running all configured services immediately", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="command")
    
    # Run
    run_parser = subparsers.add_parser("run", help="Start Language Pipes as a stdout stream without the TUI")
    run_parser.add_argument("-t", "--token", help="HuggingFace token used to download gated models")

    # Config
    subparsers.add_parser("config", help="Verify configuration file and flags")

    # Keygen
    subparsers.add_parser("keygen", help="Generate and print a new AES network key")

    return parser

def validate_config_arg(config_arg: str):
    config_path: Optional[Path] = None
    if ".toml" in config_arg:
        config_path = Path(config_arg)
    else:
        config_path = get_app_dir() / "configs" / (config_arg + ".toml")

    return os.path.exists(config_path)

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    if args.config is not None and not validate_config_arg(args.config):
            print(f"ERROR: {args.config} is not a valid path or saved configuration")
            exit()

    if args.command is None:
        from language_pipes.tui import initialize_tui

        # No console handler: stdout would corrupt the ANSI frame.
        setup_logging(console=False)
        config_args = ConfigurationArgs(args)
        initialize_tui(config_args.config_file, config_args.auto_start)
        
    elif args.command == "run":
        config_args = ConfigurationArgs(args)
        config_file = config_args.config_file
        if config_file is None:
            print("ERROR: --config param required")
            return
        
        if ".toml" in config_file and not os.path.exists(config_file):
            print(f"ERROR: {config_file} not found")
            return
        
        if ".toml" not in config_file:
            config_file = get_app_dir() / "configs" / (config_file + ".toml")
            if not os.path.exists(config_file):
                print(f"ERROR: {config_file} not found")
                return
        
        setup_logging(console=True)
        from language_pipes.runner import LpRunner
        LpRunner(Path(config_file), args.token)
        
    elif args.command == "config":
        config_args = ConfigurationArgs(args)
        config_file = config_args.config_file
        if config_file is None:
            print("ERROR: --config param required")
            return
        
        if ".toml" in config_file and not os.path.exists(config_file):
            print(f"ERROR: {config_file} not found")
            return
        
        if ".toml" not in config_file:
            config_file = get_app_dir() / "configs" / (config_file + ".toml")
            if not os.path.exists(config_file):
                print(f"ERROR: {config_file} not found")
                return

        from language_pipes.config import LpConfig
        config = LpConfig.from_file(Path(config_file))
        print(config.to_string())
        
    elif args.command == "keygen":
        key = generate_aes_key().hex()
        print(f"✓ Network key generated: {key}")
        
if __name__ == "__main__":
    main()
