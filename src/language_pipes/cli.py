import argparse
from importlib import resources
import os
from pathlib import Path
from typing import Optional
from language_pipes.config_args import ConfigurationArgs
from language_pipes.util.config import get_app_dir

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
    parser.add_argument("-c", "config", help="Load configuration from TOML file")
    parser.add_argument("-s", "start", help="Start running all configured services immediately", default=False)

    subparsers = parser.add_subparsers(dest="command")
    
    # Run
    run_parser = subparsers.add_parser("run", help="Start Language Pipes as a stdout stream without the TUI")
    
    run_parser.add_argument("-c", "config", help="Load configuration from TOML file", required=True)
    run_parser.add_argument("-s", "set", help="Override config property by its TOML key name. Repeatable", dest="set_overrides", action="append", metavar="KEY=VALUE")

    run_parser.add_argument("layer-models", help="Models to host", nargs="*", metavar="MODEL")
    run_parser.add_argument("end-models", help="Model IDs for which to load end models", nargs="*", metavar="END")

    # Config
    config_parser = subparsers.add_parser("config", help="Verify configuration file and flags")
    config_parser.add_argument("-c", "config", help="Configuration file to resolve")
    config_parser.add_argument("-s", "set", help="Override a property (same as run)", dest="set_overrides", action="append", metavar="KEY=VALUE")
    
    config_parser.add_argument("layer-models", help="Models to host", nargs="*", metavar="MODEL")
    config_parser.add_argument("end-models", help="Model IDs for which to load end models", nargs="*", metavar="END")
    
    # Keygen
    keygen_parser = subparsers.add_parser("keygen", help="Generate AES encryption key")
    keygen_parser.add_argument(
        "output",
        nargs="?",
        help="Output file for AES key (default: network.key)",
        default="network.key",
    )

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
        initialize_tui(args.config, getattr(args, "start", False))
        
    elif args.command == "run":
        config_args = ConfigurationArgs(args)
        # Start headless
        pass
    elif args.command == "config":
        config_args = ConfigurationArgs(args)
        # Validate config
        pass

if __name__ == "__main__":
    main()
