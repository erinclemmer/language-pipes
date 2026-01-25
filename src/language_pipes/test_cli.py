# Allows testing of cli by using `python src/langugage_pipes/test_cli.py` when at the repository base directory

import os
import sys
import pathlib
cd = pathlib.Path().resolve()
sys.path.append(os.path.join(cd, 'src'))

from language_pipes.cli import main

main(sys.argv[1:])
