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

    args = parser.parse_args()

    if args.command == "create_key":
        with open(args.output_file, 'wb') as f:
            f.write(generate_aes_key())
    elif args.command == "run":
        LanguagePipes(version, config_file=args.config_file)
    else:
        parser.print_usage()
        exit(1)

if __name__ == "__main__":
    main()
