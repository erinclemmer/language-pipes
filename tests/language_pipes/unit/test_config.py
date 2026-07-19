import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.util.config import (
    get_max_node_jobs,
    get_max_api_jobs,
    is_8_bit_mode,
    initialize_folders,
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


class InitializeFoldersTests(unittest.TestCase):
    def test_creates_app_model_config_and_credential_dirs(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as model_dir:
            app_path = Path(app_dir) / "app"
            model_path = Path(model_dir) / "models"
            env = {"LP_APP_DIR": str(app_path), "LP_MODEL_DIR": str(model_path)}
            with mock.patch.dict(os.environ, env, clear=True):
                initialize_folders()

            self.assertTrue(app_path.is_dir())
            self.assertTrue(model_path.is_dir())
            self.assertTrue((app_path / "configs").is_dir())
            self.assertTrue((app_path / "credentials").is_dir())

    def test_is_idempotent_when_dirs_already_exist(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as model_dir:
            env = {"LP_APP_DIR": app_dir, "LP_MODEL_DIR": model_dir}
            with mock.patch.dict(os.environ, env, clear=True):
                initialize_folders()
                # Second call must not raise even though everything already exists.
                initialize_folders()

            self.assertTrue((Path(app_dir) / "configs").is_dir())
            self.assertTrue((Path(app_dir) / "credentials").is_dir())


if __name__ == "__main__":
    unittest.main()
