"""Unit tests for language_pipes.

Run all tests with: python -m tests.language_pipes.unit
"""
import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


def load_tests(loader, tests, pattern):
    """Load all tests from this package."""
    package_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir=package_dir, pattern='test_*.py')
    return suite


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = load_tests(loader, None, None)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
