"""Unit test package entrypoint for discovery."""

from __future__ import annotations

from pathlib import Path
import unittest


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    return loader.discover(start_dir=Path(__file__).parent, pattern="test_*.py")


if __name__ == "__main__":
    unittest.main()
