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

    if len(sys.argv) < 2:
        print("Usage: language_pipes [config_file]")
        exit()

    if sys.argv[1] == 'create_key':
        if len(sys.argv) < 3:
            print("Usage: language_pipes create_key [output_file]")
            exit()
        with open(sys.argv[2], 'wb') as f:
            f.write(generate_aes_key())
        exit()

    LanguagePipes(version, config_file=sys.argv[1])

if __name__ == "__main__":
    main()
