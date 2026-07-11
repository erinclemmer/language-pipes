import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.util.config import (
    get_max_node_jobs,
    get_max_api_jobs,
    is_8_bit_mode,
)


class MaxNodeJobsTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_ten(self):
        self.assertEqual(get_max_node_jobs(), 10)

    @mock.patch.dict(os.environ, {"LP_MAX_NODE_JOBS": "3"}, clear=True)
    def test_reads_env_override(self):
        self.assertEqual(get_max_node_jobs(), 3)


class MaxApiJobsTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_five(self):
        self.assertEqual(get_max_api_jobs(), 5)

    @mock.patch.dict(os.environ, {"LP_MAX_API_JOBS": "1"}, clear=True)
    def test_reads_env_override(self):
        self.assertEqual(get_max_api_jobs(), 1)


class EightBitModeTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_false(self):
        self.assertFalse(is_8_bit_mode())

    @mock.patch.dict(os.environ, {"LP_8_BIT_MODE": "false"}, clear=True)
    def test_explicit_false(self):
        self.assertFalse(is_8_bit_mode())

    def test_truthy_values_enable(self):
        for value in ("true", "True", "TRUE", "1", "yes", "YES"):
            with mock.patch.dict(os.environ, {"LP_8_BIT_MODE": value}, clear=True):
                self.assertTrue(is_8_bit_mode(), f"{value!r} should enable 8-bit mode")

    def test_non_truthy_values_disable(self):
        for value in ("0", "no", "off", "", "enabled"):
            with mock.patch.dict(os.environ, {"LP_8_BIT_MODE": value}, clear=True):
                self.assertFalse(is_8_bit_mode(), f"{value!r} should not enable 8-bit mode")


if __name__ == "__main__":
    unittest.main()
