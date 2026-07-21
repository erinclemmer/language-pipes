"""Tests for LpRunner (src/language_pipes/runner.py).

Covers the behaviour documented in the 2.4.0-dev release notes:
- folder initialization on startup
- generating a new ECDSA node id if the configured one doesn't exist yet
- downloading any configured models that aren't installed, falling back to
  the globally configured HuggingFace token when none is passed explicitly
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.config import LpConfig, ModelToLoad, EndModelConfig
from language_pipes.runner import LpRunner


def make_model(model_id: str) -> ModelToLoad:
    return ModelToLoad(model_id=model_id, device=torch.device("cpu"), memory=1.0)


def make_config(node_id="node-a", layer_models=None, end_models=None):
    cfg = LpConfig()
    cfg.network_config.node_id = node_id
    cfg.layer_models = layer_models or []
    cfg.end_models = [
        m if isinstance(m, EndModelConfig) else EndModelConfig(model_id=m)
        for m in (end_models or [])
    ]
    return cfg


def make_bare_runner(provider):
    """Build an LpRunner without running __init__ (which spins up networking)."""
    runner = LpRunner.__new__(LpRunner)
    runner.provider = provider
    return runner


class GenerateNodeIdTests(unittest.TestCase):
    def test_saves_new_node_id_when_missing(self):
        provider = mock.MagicMock()
        provider.network_provider.get_my_node_ids.return_value = ["some-other-node"]
        runner = make_bare_runner(provider)

        runner._generate_node_id(make_config(node_id="node-a"))

        provider.network_provider.save_new_node_id.assert_called_once_with("node-a")

    def test_does_not_regenerate_existing_node_id(self):
        provider = mock.MagicMock()
        provider.network_provider.get_my_node_ids.return_value = ["node-a", "node-b"]
        runner = make_bare_runner(provider)

        runner._generate_node_id(make_config(node_id="node-a"))

        provider.network_provider.save_new_node_id.assert_not_called()


class DownloadModelsTests(unittest.TestCase):
    def test_skips_already_installed_models(self):
        provider = mock.MagicMock()
        provider.model_provider.get_installed_models.return_value = ["org/layer-model", "org/end-model"]
        runner = make_bare_runner(provider)
        runner._download_model = mock.MagicMock()

        cfg = make_config(
            layer_models=[make_model("org/layer-model")],
            end_models=["org/end-model"],
        )
        runner._download_models(cfg, "explicit-token")

        runner._download_model.assert_not_called()

    def test_downloads_missing_layer_and_end_models(self):
        provider = mock.MagicMock()
        provider.model_provider.get_installed_models.return_value = []
        runner = make_bare_runner(provider)
        runner._download_model = mock.MagicMock()

        cfg = make_config(
            layer_models=[make_model("org/layer-model")],
            end_models=["org/end-model"],
        )
        runner._download_models(cfg, "explicit-token")

        runner._download_model.assert_any_call("org/layer-model", "explicit-token")
        runner._download_model.assert_any_call("org/end-model", "explicit-token")
        self.assertEqual(runner._download_model.call_count, 2)

    def test_uses_explicit_token_over_global_config(self):
        provider = mock.MagicMock()
        provider.model_provider.get_installed_models.return_value = []
        runner = make_bare_runner(provider)
        runner._download_model = mock.MagicMock()

        cfg = make_config(layer_models=[make_model("org/m")])
        runner._download_models(cfg, "explicit-token")

        runner._download_model.assert_called_once_with("org/m", "explicit-token")
        provider.model_provider.get_hf_config_token.assert_not_called()

    def test_falls_back_to_global_hf_token_when_none_passed(self):
        provider = mock.MagicMock()
        provider.model_provider.get_installed_models.return_value = []
        provider.model_provider.get_hf_config_token.return_value = "global-token"
        runner = make_bare_runner(provider)
        runner._download_model = mock.MagicMock()

        cfg = make_config(layer_models=[make_model("org/m")])
        runner._download_models(cfg, None)

        provider.model_provider.get_hf_config_token.assert_called_once()
        runner._download_model.assert_called_once_with("org/m", "global-token")

    def test_does_not_download_same_model_twice(self):
        # A model_id appearing in both layer_models and end_models should
        # only be downloaded once.
        provider = mock.MagicMock()
        provider.model_provider.get_installed_models.return_value = []
        runner = make_bare_runner(provider)
        runner._download_model = mock.MagicMock()

        cfg = make_config(
            layer_models=[make_model("org/shared")],
            end_models=["org/shared"],
        )
        runner._download_models(cfg, "token")

        self.assertEqual(runner._download_model.call_count, 1)


class DownloadModelTests(unittest.TestCase):
    def test_starts_download_and_waits_for_thread(self):
        provider = mock.MagicMock()
        thread = mock.MagicMock()
        provider.model_provider.download_model_thread = thread
        provider.model_provider.download_message = "[SUCCESS] Download complete"
        runner = make_bare_runner(provider)

        runner._download_model("org/m", "tok")

        provider.model_provider.start_download.assert_called_once_with("org/m", "tok")
        thread.join.assert_called_once()

    def test_logs_error_on_failed_download(self):
        provider = mock.MagicMock()
        provider.model_provider.download_model_thread = None
        provider.model_provider.download_message = "[ERROR] Repository not found"
        runner = make_bare_runner(provider)

        with self.assertLogs("language_pipes.runner", level="ERROR") as logs:
            runner._download_model("org/missing", None)

        self.assertTrue(any("Failed to download org/missing" in m for m in logs.output))


class RunnerInitTests(unittest.TestCase):
    """Exercises the full startup sequence with all IO/network dependencies mocked."""

    def _run_init(self, node_ids, installed_models, hf_token=None, cli_token=None):
        provider = mock.MagicMock()
        provider.network_provider.get_my_node_ids.return_value = node_ids
        provider.network_provider.router_starting = False
        provider.model_provider.get_installed_models.return_value = installed_models
        provider.model_provider.get_hf_config_token.return_value = hf_token
        provider.model_provider.download_model_thread = None
        provider.model_provider.download_message = "[SUCCESS] Download complete"

        with mock.patch("language_pipes.runner.initialize_folders") as init_folders, \
             mock.patch("language_pipes.runner.LpConfig") as lp_config_cls, \
             mock.patch("language_pipes.runner.ContentProvider", return_value=provider), \
             mock.patch.object(LpRunner, "wait"):
            lp_config_cls.from_file.return_value = make_config(
                node_id="node-a",
                layer_models=[make_model("org/layer")],
                end_models=["org/end"],
            )
            runner = LpRunner(Path("config.toml"), cli_token)

        return runner, provider, init_folders

    def test_initializes_folders_before_anything_else(self):
        _, _, init_folders = self._run_init(node_ids=["node-a"], installed_models=["org/layer", "org/end"])
        init_folders.assert_called_once()

    def test_generates_node_id_when_missing(self):
        _, provider, _ = self._run_init(node_ids=[], installed_models=["org/layer", "org/end"])
        provider.network_provider.save_new_node_id.assert_called_once_with("node-a")

    def test_downloads_missing_models_with_explicit_token(self):
        _, provider, _ = self._run_init(node_ids=["node-a"], installed_models=[], cli_token="explicit")
        calls = [c.args for c in provider.model_provider.start_download.call_args_list]
        self.assertIn(("org/layer", "explicit"), calls)
        self.assertIn(("org/end", "explicit"), calls)

    def test_starts_network_and_loads_models(self):
        _, provider, _ = self._run_init(node_ids=["node-a"], installed_models=["org/layer", "org/end"])
        provider.network_provider.start_network.assert_called_once()
        provider.job_provider.start_oai_server.assert_called_once()
        provider.model_provider.load_layer_model.assert_called_once()
        provider.model_provider.load_end_model.assert_called_once_with("org/end")


if __name__ == "__main__":
    unittest.main()
