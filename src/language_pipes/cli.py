import os
import socket
import toml
import argparse
from language_pipes.config import LpConfig
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes.commands.start import start_wizard
from language_pipes.commands.upgrade import upgrade_lp

from language_pipes import LanguagePipes

VERSION = "0.12.2"

def build_parser():
    parser = argparse.ArgumentParser(
        prog="Language Pipes",
        description="Distribute LLMs across multiple systems"
    )

    parser.add_argument("-V", "--version", action="version", version=VERSION)

    subparsers = parser.add_subparsers(dest="command")

    #Upgrade
    subparsers.add_parser("upgrade", help="Upgrade Language Pipes and related packages")

    # Key Generation
    create_key_parser = subparsers.add_parser("keygen", help="Generate AES key")
    create_key_parser.add_argument("output", help="Output file for AES key (default: network.key)", default="network.key")

    # Initialize
    init = subparsers.add_parser("init", help="Create a new configuration file")
    init.add_argument("-o", "--output", default="config.toml")

    # Quick start
    start = subparsers.add_parser("start", help="First-time setup wizard and server start")
    start.add_argument("-c", "--config", default="config.toml", help="Config file path")
    start.add_argument("-k", "--key", default="network.key", help="Network key file path")

    # run command
    run_parser = subparsers.add_parser("serve", help="Start Language Pipes server")
    run_parser.add_argument("-c", "--config", help="Path to TOML config file")
    run_parser.add_argument("-l", "--logging-level", 
        help="Logging verbosity (Default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    run_parser.add_argument("--openai-port", type=int, help="Open AI server port (Default: none)")
    run_parser.add_argument("-d", "--app-data-dir", type=str, help="Application data directory for language pipes (default: ~/.language-pipes)")
    run_parser.add_argument("--node-id", help="Node ID for the network (Required)")
    run_parser.add_argument("--app-dir", type=str, help="Directory to store data for this application")
    run_parser.add_argument("--peer-port", type=int, help="Port for peer-to-peer network (Default: 5000)")
    run_parser.add_argument("--bootstrap-address", help="Bootstrap node address (e.g. 192.168.1.100)")
    run_parser.add_argument("--bootstrap-port", type=int, help="Bootstrap node port for the network (e.g. 8000)")
    run_parser.add_argument("--max-pipes", type=int, help="Maximum amount of pipes to host")
    run_parser.add_argument("--network-key", type=str, help="AES key to access network (Default: network.key)")
    run_parser.add_argument("--model-validation", help="Whether to validate the model weight hashes when connecting to a pipe.", action="store_true")
    run_parser.add_argument("--ecdsa-verification", help="verify legitimacy of sender via ecdsa signed packets" , action="store_true")
    run_parser.add_argument("--job-port", type=int, help="Job receiver port (Default: 5050)")
    run_parser.add_argument("--hosted-models", nargs="*", metavar="MODEL", 
        help="Hosted models as key=value pairs: id=MODEL,device=DEVICE,memory=GB,load_ends=BOOL (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=false)")

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
        "ecdsa_verification": os.getenv("LP_ECDSA_VERIFICATION"),
        "model_validation": os.getenv("LP_MODEL_VALIDATION"),
        "job_port": os.getenv("LP_JOB_PORT"),
        "max_pipes": os.getenv("LP_MAX_PIPES"),
        "hosted_models": os.getenv("LP_HOSTED_MODELS"),
    }

    def precedence(key, arg, d):
        if arg is not None:
            return arg
        if key in env_map and env_map[key] is not None:
            return env_map[key]
        if key in data:
            return data[key]
        return d
    
    default_app_dir = os.path.expanduser("~") + "/.language-pipes"

    config = {
        "logging_level": precedence("logging_level", args.logging_level, "INFO"),
        "oai_port": precedence("oai_port", args.openai_port, None),
        "app_dir": precedence("app_dir", args.app_dir, default_app_dir),
        "node_id": precedence("node_id", args.node_id, None),
        "peer_port": int(precedence("peer_port", args.peer_port, 5000)),
        "network_ip": precedence("network_ip", None, "127.0.0.1"),
        "bootstrap_address": precedence("bootstrap_address", args.bootstrap_address, None),
        "bootstrap_port": precedence("bootstrap_port", args.bootstrap_port, 5000),
        "network_key": precedence("network_key", args.network_key, None),
        "ecdsa_verification": precedence("ecdsa_verification", args.ecdsa_verification, False),
        "model_validation": precedence("model_validation", args.model_validation, False),
        "job_port": int(precedence("job_port", args.job_port, 5050)),
        "max_pipes": precedence("max_pipes", args.max_pipes, 1),
        "hosted_models": precedence("hosted_models", args.hosted_models, None),
    }

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
        config = LpConfig.from_dict({
            "logging_level": data["logging_level"],
            "oai_port": data["oai_port"],
            "app_dir": data["app_dir"],
            "router": {
                "node_id": data["node_id"],
                "port": data["peer_port"],
                "network_ip": data["network_ip"],
                "aes_key": data["network_key"],
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

        return LanguagePipes(VERSION, config)
    else:
        parser.print_usage()
        exit(1)

if __name__ == "__main__":
    main()
