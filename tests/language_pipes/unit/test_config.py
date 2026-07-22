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
from language_pipes.config import LpConfig, EndModelConfig, DEFAULT_NUM_LOCAL_LAYERS


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


class EndModelConfigTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_string_form_defaults_num_local_layers(self):
        cfg = EndModelConfig.from_config("org/model")
        self.assertEqual(cfg.model_id, "org/model")
        self.assertEqual(cfg.num_local_layers, DEFAULT_NUM_LOCAL_LAYERS)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_object_form_reads_num_local_layers(self):
        cfg = EndModelConfig.from_config({"model_id": "org/model", "num_local_layers": 3})
        self.assertEqual(cfg.model_id, "org/model")
        self.assertEqual(cfg.num_local_layers, 3)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_object_form_without_num_local_layers_uses_default(self):
        cfg = EndModelConfig.from_config({"model_id": "org/model"})
        self.assertEqual(cfg.num_local_layers, DEFAULT_NUM_LOCAL_LAYERS)

    def test_to_config_uses_string_when_default(self):
        cfg = EndModelConfig(model_id="org/model")
        self.assertEqual(cfg.to_config(), "org/model")

    def test_to_config_uses_object_when_non_default(self):
        cfg = EndModelConfig(model_id="org/model", num_local_layers=2)
        self.assertEqual(
            cfg.to_config(),
            {"model_id": "org/model", "num_local_layers": 2, "device": "cpu"},
        )

    def test_to_config_uses_object_when_non_default_device(self):
        cfg = EndModelConfig(model_id="org/model", device="cuda:0")
        self.assertEqual(
            cfg.to_config(),
            {
                "model_id": "org/model",
                "num_local_layers": DEFAULT_NUM_LOCAL_LAYERS,
                "device": "cuda:0",
            },
        )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_object_form_reads_device(self):
        cfg = EndModelConfig.from_config(
            {"model_id": "org/model", "device": "cuda:1"}
        )
        self.assertEqual(cfg.device, "cuda:1")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_object_form_without_device_uses_default(self):
        cfg = EndModelConfig.from_config({"model_id": "org/model"})
        self.assertEqual(cfg.device, "cpu")

    @mock.patch.dict(os.environ, {"LP_NUM_LOCAL_LAYERS": "4"}, clear=True)
    def test_deprecated_env_var_used_as_fallback_default(self):
        cfg = EndModelConfig.from_config("org/model")
        self.assertEqual(cfg.num_local_layers, 4)

    @mock.patch.dict(os.environ, {"LP_NUM_LOCAL_LAYERS": "4"}, clear=True)
    def test_explicit_num_local_layers_overrides_env_var(self):
        cfg = EndModelConfig.from_config({"model_id": "org/model", "num_local_layers": 1})
        self.assertEqual(cfg.num_local_layers, 1)

    @mock.patch.dict(os.environ, {"LP_NUM_LOCAL_LAYERS": "not-an-int"}, clear=True)
    def test_invalid_env_var_falls_back_to_default(self):
        cfg = EndModelConfig.from_config("org/model")
        self.assertEqual(cfg.num_local_layers, DEFAULT_NUM_LOCAL_LAYERS)


class EndModelsRoundTripTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_mixed_forms_survive_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            cfg = LpConfig()
            cfg._file_path = path
            cfg.end_models = [
                EndModelConfig(model_id="org/simple"),
                EndModelConfig(model_id="org/multi", num_local_layers=3),
            ]
            cfg.save()

            reloaded = LpConfig.from_file(path)

            self.assertEqual(len(reloaded.end_models), 2)
            self.assertEqual(reloaded.end_models[0].model_id, "org/simple")
            self.assertEqual(reloaded.end_models[0].num_local_layers, DEFAULT_NUM_LOCAL_LAYERS)
            self.assertEqual(reloaded.end_models[1].model_id, "org/multi")
            self.assertEqual(reloaded.end_models[1].num_local_layers, 3)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_legacy_string_list_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            import toml
            with open(path, "w", encoding="utf-8") as f:
                toml.dump({"end_models": ["org/a", "org/b"]}, f)

            cfg = LpConfig.from_file(path)

            self.assertEqual([m.model_id for m in cfg.end_models], ["org/a", "org/b"])
            self.assertTrue(all(m.num_local_layers == DEFAULT_NUM_LOCAL_LAYERS for m in cfg.end_models))


if __name__ == "__main__":
    unittest.main()
