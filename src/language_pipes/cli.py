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

def main():
    version = get_version()

    parser = argparse.ArgumentParser(description="Language Pipes CLI")
    subparsers = parser.add_subparsers(dest="command")

    # create_key command
    create_key_parser = subparsers.add_parser("create_key", help="Generate AES key")
    create_key_parser.add_argument("output_file", help="Output file for AES key")

    # run command (default)
    run_parser = subparsers.add_parser("run", help="Run Language Pipes with config")
    run_parser.add_argument("config_file", help="Path to TOML config file")

    # Common overrides for run
    parser.add_argument("--logging-level", help="Override logging level")
    parser.add_argument("--oai-port", type=int, help="Override OAI port")
    parser.add_argument("--router-host", help="Override router host")
    parser.add_argument("--router-port", type=int, help="Override router port")
    parser.add_argument("--https", type=bool, help="Override HTTPS setting")
    parser.add_argument("--job-port", type=int, help="Override job port")
    parser.add_argument("--hosted-models", nargs="*", help="Override hosted models in format id[:device[:max_memory]]")

    args = parser.parse_args()

    if args.command == "create_key":
        with open(args.output_file, 'wb') as f:
            f.write(generate_aes_key())
    elif args.command == "run":
        from language_pipes.config import LMNetConfig
        config = LMNetConfig.from_file(args.config_file)

        # Apply overrides
        if args.logging_level:
            config.logging_level = args.logging_level
        if args.oai_port:
            config.oai_port = args.oai_port
        if args.router_host:
            config.router.host = args.router_host
        if args.router_port:
            config.router.port = args.router_port
        if args.https is not None:
            config.processor.https = args.https
        if args.job_port:
            config.processor.job_port = args.job_port
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

        LanguagePipes(version, config_file=args.config_file)
    else:
        parser.print_usage()
        exit(1)

if __name__ == "__main__":
    main()
