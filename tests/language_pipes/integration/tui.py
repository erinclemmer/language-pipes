import os
import sys
import pathlib
import unittest

cd = pathlib.Path().resolve()

sys.path.append(os.path.join(cd, 'src'))

from language_pipes.cli import main


class OpenAITests(unittest.TestCase):
    def test_cli(self):
        main([])

    def test_tui(self):
        main(["tui"])

if __name__ == '__main__':
    unittest.main()
