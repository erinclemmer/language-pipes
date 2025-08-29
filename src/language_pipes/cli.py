import os
import sys
import argparse
from language_pipes.util.aes import generate_aes_key

from __init__ import LanguagePipes

def get_version():
    if not os.path.exists("release.txt"):
        raise Exception("Could not find release.txt file")
    with open("release.txt", "r", encoding="utf-8") as f:
        version = f.read()

def build_parser():
    parser = argparse.ArgumentParser(description="Language Pipes CLI")
    subparsers = parser.add_subparsers(dest="command")

    # create_key command
    create_key_parser = subparsers.add_parser("create_key", help="Generate AES key")
    create_key_parser.add_argument("output_file", help="Output file for AES key")

    # run command
    run_parser = subparsers.add_parser("run", help="Run Language Pipes with config")
    run_parser.add_argument("config_file", help="Path to TOML config file")
    run_parser.add_argument("--logging-level", help="Override logging level")
    run_parser.add_argument("--oai-port", type=int, help="Override OAI port")
    run_parser.add_argument("--router-host", help="Override router host")
    run_parser.add_argument("--router-port", type=int, help="Override router port")
    run_parser.add_argument("--https", type=bool, help="Override HTTPS setting")
    run_parser.add_argument("--job-port", type=int, help="Override job port")
    run_parser.add_argument("--hosted-models", nargs="*", help="Override hosted models in format id[:device[:max_memory]]")

    return parser

def apply_overrides(config, args):
    # Environment variable mapping
    env_map = {
        "logging_level": os.getenv("LP_LOGGING_LEVEL"),
        "oai_port": os.getenv("LP_OAI_PORT"),
        "router_host": os.getenv("LP_ROUTER_HOST"),
        "router_port": os.getenv("LP_ROUTER_PORT"),
        "https": os.getenv("LP_HTTPS"),
        "job_port": os.getenv("LP_JOB_PORT"),
        "hosted_models": os.getenv("LP_HOSTED_MODELS"),
    }

    # Apply precedence: args > env > config > defaults
    config.logging_level = args.logging_level or env_map["logging_level"] or getattr(config, "logging_level", "INFO")
    config.oai_port = args.oai_port or (int(env_map["oai_port"]) if env_map["oai_port"] else None) or getattr(config, "oai_port", 8000)
    config.router.host = args.router_host or env_map["router_host"] or getattr(config.router, "host", "127.0.0.1")
    config.router.port = args.router_port or (int(env_map["router_port"]) if env_map["router_port"] else None) or getattr(config.router, "port", 9000)
    if args.https is not None:
        config.processor.https = args.https
    elif env_map["https"] is not None:
        config.processor.https = env_map["https"].lower() in ("1", "true", "yes")
    else:
        config.processor.https = getattr(config.processor, "https", False)
    config.processor.job_port = args.job_port or (int(env_map["job_port"]) if env_map["job_port"] else None) or getattr(config.processor, "job_port", 10000)

    # Hosted models are required
    hosted_models_arg = args.hosted_models
    hosted_models_env = env_map["hosted_models"].split() if env_map["hosted_models"] else None
    if hosted_models_arg or hosted_models_env:
        from language_pipes.config.processor import HostedModel
        models = []
        for m in (hosted_models_arg or hosted_models_env):
            parts = m.split(":")
            id = parts[0]
            device = parts[1] if len(parts) > 1 else "cpu"
            max_memory = int(parts[2]) * 10**9 if len(parts) > 2 else 0
            models.append(HostedModel(id, device, max_memory))
        config.processor.hosted_models = models
    elif not getattr(config.processor, "hosted_models", None):
        raise ValueError("At least one hosted_model must be specified via config, CLI, or environment")

    return config

def main():
    version = get_version()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create_key":
        with open(args.output_file, 'wb') as f:
            f.write(generate_aes_key())
    elif args.command == "run":
        from language_pipes.config import LMNetConfig
        config = LMNetConfig.from_file(args.config_file)
        config = apply_overrides(config, args)
        LanguagePipes(version, config_file=args.config_file)
    else:
        parser.print_usage()
        exit(1)

if __name__ == "__main__":
    main()
