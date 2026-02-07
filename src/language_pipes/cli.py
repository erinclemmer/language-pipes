import toml
import argparse

from language_pipes.distributed_state_network import DSNodeConfig, DSNodeServer

from language_pipes.config import LpConfig, apply_env_overrides
from language_pipes.util.aes import save_new_aes_key
from language_pipes.commands.initialize import interactive_init
from language_pipes.commands.start import start_wizard
from language_pipes.commands.upgrade import upgrade_lp

from language_pipes.lp import LanguagePipes

VERSION = "0.19.7"

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
    run_parser.add_argument("--layer-models", nargs="*", metavar="MODEL", 
        help="Layer models as key=value pairs: id=MODEL,device=DEVICE,memory=GB (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4)")
    run_parser.add_argument("--end-models", nargs="*", metavar="END", help="End models to host as model IDs like \"Qwen/Qwen3-1.7B\"")
    run_parser.add_argument("--num-local-layers", type=int, help="Number of local layers to run on your machine. More layers means better prompt obfuscation")
    run_parser.add_argument("--prefill-chunk-size", help="Number of tokens to process for each batch in prefill", type=int)

    return parser

def apply_overrides(data, args):
    app_dir_arg = args.app_dir
    if app_dir_arg is None and hasattr(args, "app_data_dir"):
        app_dir_arg = args.app_data_dir

    cli_args = {
        "logging_level": args.logging_level,
        "openai_port": args.openai_port,
        "app_dir": app_dir_arg,
        "node_id": args.node_id,
        "peer_port": args.peer_port,
        "bootstrap_address": args.bootstrap_address,
        "bootstrap_port": args.bootstrap_port,
        "network_key": args.network_key,
        "model_validation": args.model_validation,
        "max_pipes": args.max_pipes,
        "layer_models": args.layer_models,
        "end_models": args.end_models,
        "prefill_chunk_size": args.prefill_chunk_size,
        "num_local_layers": args.num_local_layers,
        "model_dir": args.model_dir,
    }

    config = apply_env_overrides(data, cli_args)

    if config["layer_models"] is None:
        print("Error: layer_models param must be supplied in config")
        exit()

    if config["node_id"] is None:
        print("Error: node_id param is not supplied in config")
        exit()

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
