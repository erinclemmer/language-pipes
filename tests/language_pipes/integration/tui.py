import os
import sys
import pathlib
import unittest

cd = pathlib.Path().resolve()

sys.path.append(os.path.join(cd, 'src'))

from language_pipes.cli import main

class OpenAITests(unittest.TestCase):
    def test_main_menu(self):
        main([])

    def test_config(self):
        main([])

    def test_run(self):
        main(["--config", "test", "run"])

if __name__ == '__main__':
    unittest.main()
