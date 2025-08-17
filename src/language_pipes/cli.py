import os
import sys

from __init__ import LanguagePipes

def main():

    if not os.path.exists("release.txt"):
        raise Exception("Could not find release.txt file")
    with open("release.txt", "r", encoding="utf-8") as f:
        version = f.read()

    if len(sys.argv) < 2:
        print("Usage: python main.py [config_file]")
        exit()

    LanguagePipes(version, config_file=sys.argv[1])

if __name__ == "__main__":
    main()
