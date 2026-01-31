import os
import toml
import argparse

from distributed_state_network import DSNodeConfig, DSNodeServer

from language_pipes.config import LpConfig
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes.commands.start import start_wizard
from language_pipes.commands.upgrade import upgrade_lp

from language_pipes.lp import LanguagePipes

VERSION = "0.19.1"

def build_parser():
    parser = argparse.ArgumentParser(
        prog="Language Pipes",
        description="Distribute LLMs across multiple systems"
    )

    parser.add_argument("-v", "--version", action="version", version=VERSION)

    subparsers = parser.add_subparsers(dest="command")

    #Upgrade
    subparsers.add_parser("upgrade", help="Upgrade Language Pipes package")

    # Key Generation
    create_key_parser = subparsers.add_parser("keygen", help="Generate AES key")
    create_key_parser.add_argument("output", nargs='?', help="Output file for AES key (default: network.key)", default="network.key")

    # Initialize
    init = subparsers.add_parser("init", help="Create a new configuration file")
    init.add_argument("output", nargs='?', default="config.toml", help="Output file name to write to (default: config.toml)")

    # run command
    run_parser = subparsers.add_parser("serve", help="Start Language Pipes server")
    run_parser.add_argument("-c", "--config", help="Path to TOML config file")
    run_parser.add_argument("-l", "--logging-level", 
        help="Logging verbosity (Default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    run_parser.add_argument("--openai-port", type=int, help="Open AI server port (Default: none)")
    run_parser.add_argument("--app-data-dir", type=str, help="Application data directory for language pipes (default: ~/.config/language_pipes)")
    run_parser.add_argument("--model-dir", type=str, help="Directory to store model data (default: ~/.cache/language_pipes/models)")
    run_parser.add_argument("--node-id", help="Node ID for the network (Required)")
    run_parser.add_argument("--app-dir", type=str, help="Directory to store data for this application")
    run_parser.add_argument("--peer-port", type=int, help="Port for peer-to-peer network (Default: 5000)")
    run_parser.add_argument("--bootstrap-address", help="Bootstrap node address (e.g. 192.168.1.100)")
    run_parser.add_argument("--bootstrap-port", type=int, help="Bootstrap node port for the network (e.g. 8000)")
    run_parser.add_argument("--max-pipes", type=int, help="Maximum amount of pipes to host")
    run_parser.add_argument("--network-key", type=str, help="AES key to access network (Default: network.key)")
    run_parser.add_argument("--model-validation", help="Whether to validate the model weight hashes when connecting to a pipe.", action="store_true")
    run_parser.add_argument("--hosted-models", nargs="*", metavar="MODEL", 
        help="Hosted models as key=value pairs: id=MODEL,device=DEVICE,memory=GB,load_ends=BOOL (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=false)")
    run_parser.add_argument("--prefill-chunk-size", help="Number of tokens to process for each batch in prefill", type=int)

    return parser

def apply_overrides(data, args):
    # Environment variable mapping
    env_map = {
        "logging_level": os.getenv("LP_LOGGING_LEVEL"),
        "oai_port": os.getenv("LP_OAI_PORT"),
        "app_dir": os.getenv("LP_APP_DIR"),
        "node_id": os.getenv("LP_NODE_ID"),
        "peer_port": os.getenv("LP_PEER_PORT"),
        "network_ip": os.getenv("LP_NETWORK_IP"),
        "bootstrap_address": os.getenv("LP_BOOTSTRAP_ADDRESS"),
        "bootstrap_port": os.getenv("LP_BOOTSTRAP_PORT"),
        "network_key": os.getenv("LP_NETWORK_KEY"),
        "model_validation": os.getenv("LP_MODEL_VALIDATION"),
        "max_pipes": os.getenv("LP_MAX_PIPES"),
        "hosted_models": os.getenv("LP_HOSTED_MODELS"),
        "prefill_chunk_size": os.getenv("LP_PREFILL_CHUNK_SIZE"),
        "model_dir": os.getenv("LP_MODEL_DIR"),
    }

    def precedence(key, arg):
        if arg is not None:
            return arg
        if key in env_map and env_map[key] is not None:
            return env_map[key]
        if key in data:
            return data[key]
        return None

    app_dir_arg = args.app_dir
    if app_dir_arg is None and hasattr(args, "app_data_dir"):
        app_dir_arg = args.app_data_dir

    config = {
        "logging_level": precedence("logging_level", args.logging_level),
        "oai_port": precedence("oai_port", args.openai_port),
        "app_dir": precedence("app_dir", app_dir_arg),
        "node_id": precedence("node_id", args.node_id),
        "peer_port": precedence("peer_port", args.peer_port),
        "network_ip": precedence("network_ip", None),
        "bootstrap_address": precedence("bootstrap_address", args.bootstrap_address),
        "bootstrap_port": precedence("bootstrap_port", args.bootstrap_port),
        "network_key": precedence("network_key", args.network_key),
        "model_validation": precedence("model_validation", args.model_validation),
        "max_pipes": precedence("max_pipes", args.max_pipes),
        "hosted_models": precedence("hosted_models", args.hosted_models),
        "prefill_chunk_size": precedence("prefill_chunk_size", args.prefill_chunk_size),
        "model_dir": precedence("model_dir", args.model_dir),
    }

    if config["peer_port"] is not None:
        config["peer_port"] = int(config["peer_port"])

    if config["hosted_models"] is None:
        print("Error: hosted_models param must be supplied in config")
        exit()

    if config["node_id"] is None:
        print("Error: node_id param is not supplied in config")
        exit()
    
    if config["oai_port"] is not None:
        config["oai_port"] = int(config["oai_port"])
    
    if config["bootstrap_port"] is not None:
        config["bootstrap_port"] = int(config["bootstrap_port"])

    hosted_models = []
    for m in config["hosted_models"]:
        if type(m) is str:
            # Parse Docker-style key=value pairs: id=X,device=Y,memory=Z,load_ends=W
            model_config = {}
            for pair in m.split(","):
                if "=" not in pair:
                    raise ValueError(f"Invalid format '{pair}' in '{m}'. Expected key=value pairs (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=false)")
                key, value = pair.split("=", 1)
                model_config[key.strip()] = value.strip()
            
            # Validate required keys
            required_keys = {"id", "device", "memory"}
            missing = required_keys - set(model_config.keys())
            if missing:
                raise ValueError(f"Missing required keys {missing} in '{m}'")
            
            hosted_models.append({
                "id": model_config["id"],
                "device": model_config["device"],
                "max_memory": float(model_config["memory"]),
                "load_ends": model_config.get("load_ends", "false").lower() == "true"
            })
        else:
            hosted_models.append(m)

    config["hosted_models"] = hosted_models

    return config

def main(argv = None):
    parser = build_parser()
    args = []
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    # Default to "start" command if no command given
    if args.command is None:
        args.command = "start"
        args.config = "config.toml"
        args.key = "network.key"

    if args.command == "keygen":
        key = save_new_aes_key(args.output)
        print(f"✓ Network key generated: {key}")
        print(f"✓ Network key saved to '{args.output}'")
    elif args.command == "upgrade":
        upgrade_lp()
    elif args.command == "init":
        interactive_init(args.output)
    elif args.command == "start":
        try:
            return start_wizard(apply_overrides, VERSION)
        except KeyboardInterrupt:
            exit()
    elif args.command == "serve":
        data = { }
        if args.config is not None:
            with open(args.config, "r", encoding="utf-8") as f:
                data = toml.load(f)
        data = apply_overrides(data, args)
        
        config = LpConfig.from_dict(data)

        print(config.to_string())

        router_config = DSNodeConfig.from_dict({
            "node_id": data["node_id"],
            "port": data.get("peer_port", 5000),
            "network_ip": data.get("network_ip", None),
            "aes_key": data.get("network_key", None),
            "bootstrap_nodes": [
                {
                    "address": data["bootstrap_address"],
                    "port": data["bootstrap_port"]
                }
            ] if data.get("bootstrap_address") is not None else []
        })

        print(router_config.to_string())

        router = DSNodeServer.start(router_config)

        app = LanguagePipes(config, router)
        return app
    else:
        parser.print_usage()
        exit(1)

if __name__ == "__main__":
    main()
