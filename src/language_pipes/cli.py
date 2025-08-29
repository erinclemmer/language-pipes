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
    # Apply overrides or defaults
    config.logging_level = args.logging_level or getattr(config, "logging_level", "INFO")
    config.oai_port = args.oai_port or getattr(config, "oai_port", 8000)
    config.router.host = args.router_host or getattr(config.router, "host", "127.0.0.1")
    config.router.port = args.router_port or getattr(config.router, "port", 9000)
    config.processor.https = args.https if args.https is not None else getattr(config.processor, "https", False)
    config.processor.job_port = args.job_port or getattr(config.processor, "job_port", 10000)

    # Hosted models are required
    if args.hosted_models:
        from language_pipes.config.processor import HostedModel
        models = []
        for m in args.hosted_models:
            parts = m.split(":")
            id = parts[0]
            device = parts[1] if len(parts) > 1 else "cpu"
            max_memory = int(parts[2]) * 10**9 if len(parts) > 2 else 0
            models.append(HostedModel(id, device, max_memory))
        config.processor.hosted_models = models
    elif not getattr(config.processor, "hosted_models", None):
        raise ValueError("At least one hosted_model must be specified via config or CLI")

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
