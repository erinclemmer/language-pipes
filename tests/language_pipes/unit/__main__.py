"""Run all unit tests for language_pipes.

Usage: python -m tests.language_pipes.unit
"""
from . import run_tests
import sys

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)