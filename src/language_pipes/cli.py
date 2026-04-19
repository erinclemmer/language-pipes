import toml
import argparse
from typing import List, Dict
from importlib import resources

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

class ConfigurationArgs:
    config_file: str
    set_overrides: Dict[str, str]
    layer_models: List[str]
    end_models: List[str]

    def __init__(self, args):
        self.config_file = args.config
        self.set_overrides = self.parse_set_overrides(getattr(args, "set_overrides", []))
        self.layer_models = self.parse_layer_models(getattr(args, "layer_models", []))
        self.end_models = getattr(args, "end_models", [])

    def parse_set_overrides(self, values: List[str]) -> Dict[str, str]:
        overrides = {}
        for item in values:
            key, value = item.split("=", 1)
            try:
                overrides[key] = toml.loads(f"x = {value}")["x"]
            except toml.TomlDecodeError:
                overrides[key] = value
        return overrides

    def parse_layer_models(self, layer_models: List[str]):
        result = []
        for m in layer_models:
            model_config = { }
            for pair in m.split(","):
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid format '{pair}' in '{m}'. "
                        "Expected key=value pairs (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4)"
                    )
                key, value = pair.split("=", 1)
                model_config[key.strip()] = value.strip()
            
            required_keys = {"id", "device", "memory"}
            missing = required_keys - set(model_config.keys())
            if missing:
                raise ValueError(f"Missing required keys {missing} in '{m}'")
            
            result.append({
                "id": model_config["id"],
                "device": model_config["device"],
                "memory": float(model_config["memory"])
            })
        
        return result

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    if args.command is None:
        from language_pipes.tui import initialize_tui

        initialize_tui()
        pass
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
