import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.global_config import GlobalConfig


class GlobalConfigSaveTests(unittest.TestCase):
    """Regression tests for the huggingface api key saving fix.

    Previously ``from_file`` left ``_file_path`` as ``None`` when the config
    file did not yet exist, so ``save()`` silently did nothing and the token
    was never persisted.
    """

    def test_from_file_sets_path_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"LP_APP_DIR": tmp}, clear=False):
                cfg = GlobalConfig.from_file()
                self.assertIsNotNone(cfg._file_path)
                self.assertEqual(str(cfg._file_path), os.path.join(tmp, "globals.toml"))

    def test_save_persists_token_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"LP_APP_DIR": tmp}, clear=False):
                cfg = GlobalConfig.from_file()
                cfg.hf_token = "hf_secret"
                cfg.save()

                self.assertTrue(os.path.exists(os.path.join(tmp, "globals.toml")))

                reloaded = GlobalConfig.from_file()
                self.assertEqual(reloaded.hf_token, "hf_secret")

    def test_save_round_trips_updated_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"LP_APP_DIR": tmp}, clear=False):
                first = GlobalConfig.from_file()
                first.hf_token = "hf_one"
                first.save()

                second = GlobalConfig.from_file()
                second.hf_token = "hf_two"
                second.save()

                self.assertEqual(GlobalConfig.from_file().hf_token, "hf_two")


if __name__ == "__main__":
    unittest.main()
