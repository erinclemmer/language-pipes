import os
import toml
import argparse
from language_pipes.config import LpConfig
from language_pipes.util.aes import generate_aes_key

from language_pipes import LanguagePipes

VERSION = "0.4.2"

def build_parser():
    parser = argparse.ArgumentParser(description="Language Pipes CLI")
    subparsers = parser.add_subparsers(dest="command")

    # create_key command
    create_key_parser = subparsers.add_parser("create_key", help="Generate AES key")
    create_key_parser.add_argument("output_file", help="Output file for AES key")

    # run command
    run_parser = subparsers.add_parser("run", help="Run Language Pipes with config")
    run_parser.add_argument("--config", help="Path to TOML config file")
    run_parser.add_argument("--logging-level", help="Logging level (Default: INFO)")
    run_parser.add_argument("--oai-port", type=int, help="Open AI server port (Default: none)")
    run_parser.add_argument("--node-id", help="Node ID for the network (Required)")
    run_parser.add_argument("--peer-port", type=int, help="Port for peer-to-peer network (Default: 5000)")
    run_parser.add_argument("--bootstrap-address", help="Bootstrap address for network")
    run_parser.add_argument("--bootstrap-port", type=int, help="Bootstrap port for the network")
    run_parser.add_argument("--max-pipes", type=int, help="Maximum amount of pipes to host")
    run_parser.add_argument("--network-key", type=str, help="AES key to access network (Default: network.key)")
    run_parser.add_argument("--model-validation", help="Whether to validate the model weight hashes when connecting to a pipe.", default=False, action=argparse.BooleanOptionalAction)
    run_parser.add_argument("--https", help="HTTPS job communication (Default: false)", default=False, action=argparse.BooleanOptionalAction)
    run_parser.add_argument("--ecdsa-verification", help="verify legitimacy of sender via ecdsa signed packets" , default=False, action=argparse.BooleanOptionalAction)
    run_parser.add_argument("--job-port", type=int, help="Job receiver port (Default: 5050)")
    run_parser.add_argument("--network-ip", type=str, help="IP address for the current device (only HTTPS)")
    run_parser.add_argument("--hosted-models", nargs="*", help="Hosted models in format [huggingface-id]::[device:::[max-memory] (Required)")

    return parser

def apply_overrides(data, args):
    # Environment variable mapping
    env_map = {
        "logging_level": os.getenv("LP_LOGGING_LEVEL"),
        "oai_port": os.getenv("LP_OAI_PORT"),
        "node_id": os.getenv("LP_NODE_ID"),
        "peer_port": os.getenv("LP_PEER_PORT"),
        "bootstrap_address": os.getenv("LP_BOOTSTRAP_ADDRESS"),
        "bootstrap_port": os.getenv("LP_BOOTSTRAP_PORT"),
        "network_key": os.getenv("LP_NETWORK_KEY"),
        "ecdsa_verification": os.getenv("LP_ECDSA_VERIFICATION"),
        "https": os.getenv("LP_HTTPS"),
        "model_validation": os.getenv("LP_MODEL_VALIDATION"),
        "job_port": os.getenv("LP_JOB_PORT"),
        "max_pipes": os.getenv("LP_MAX_PIPES"),
        "network_ip": os.getenv("LP_NETWORK_IP"),
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
    
    config = {
        "logging_level": precedence("logging_level", args.logging_level, "INFO"),
        "oai_port": precedence("oai_port", args.oai_port, None),
        "node_id": precedence("node_id", args.node_id, None),
        "peer_port": int(precedence("peer_port", args.peer_port, 5000)),
        "bootstrap_address": precedence("bootstrap_address", args.bootstrap_address, None),
        "bootstrap_port": precedence("bootstrap_port", args.bootstrap_port, 5000),
        "network_key": precedence("network_key", args.network_key, "network.key"),
        "https": precedence("https", args.https, False),
        "ecdsa_verification": precedence("ecdsa_verification", args.ecdsa_verification, False),
        "model_validation": precedence("model_validation", args.model_validation, False),
        "job_port": int(precedence("job_port", args.job_port, 5050)),
        "max_pipes": precedence("max_pipes", args.max_pipes, 1),
        "network_ip": precedence("network_ip", args.network_ip, "127.0.0.1"),
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
    
    config["https"] = config["https"] == "yes" or config["https"] or config["https"] == "1" or config["https"] == "true"

    hosted_models = []
    for m in config["hosted_models"]:
        if type(m) is type(''):    
            parts = m.split("::")
            if len(parts) != 3:
                raise ValueError(f"{m} is not an acceptable format for hosted_models (must be id::device::max_memory)")
            hosted_models.append({
                "id": parts[0],
                "device": parts[1],
                "max_memory": float(parts[2])
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

    if args.command == "create_key":
        with open(args.output_file, 'wb') as f:
            f.write(generate_aes_key())
    elif args.command == "run":
        data = { }
        if args.config is not None:
            with open(args.config, "r", encoding="utf-8") as f:
                data = toml.load(f)
        data = apply_overrides(data, args)
        config = LpConfig.from_dict({
            "logging_level": data["logging_level"],
            "oai_port": data["oai_port"],
            "router": {
                "node_id": data["node_id"],
                "port": data["peer_port"],
                "https": data["https"],
                "network_ip": data["network_ip"],
                "aes_key_file": data["network_key"],
                "bootstrap_nodes": [
                    {
                        "address": data["bootstrap_address"],
                        "port": data["bootstrap_port"]
                    }
                ] if data["bootstrap_address"] is not None else []
            },
            "processor": {
                "https": data["https"],
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
