import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))


def install_stub_modules():
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    sys.modules.setdefault("torch", torch)

    tqdm = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")

    class _Tqdm:
        _instances = set()

        def __init__(self, *args, **kwargs):
            self.disable = False

        @classmethod
        def get_lock(cls):
            class _Lock:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Lock()

    tqdm_auto.tqdm = _Tqdm
    sys.modules.setdefault("tqdm", tqdm)
    sys.modules.setdefault("tqdm.auto", tqdm_auto)

    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda *args, **kwargs: None
    hf.errors = types.SimpleNamespace(
        RepositoryNotFoundError=type("RepositoryNotFoundError", (Exception,), {}),
        HFValidationError=type("HFValidationError", (Exception,), {}),
    )
    sys.modules.setdefault("huggingface_hub", hf)

    model_manager = types.ModuleType("language_pipes.modeling.model_manager")

    class ModelManager:
        def __init__(self):
            self.models = []
            self.end_models = []
            self.pipes_hosted = {}

    model_manager.ModelManager = ModelManager
    sys.modules.setdefault("language_pipes.modeling.model_manager", model_manager)

    router_pipes = types.ModuleType("language_pipes.pipes.router_pipes")
    router_pipes.RouterPipes = object
    sys.modules.setdefault("language_pipes.pipes.router_pipes", router_pipes)

    dsn_util = types.ModuleType("language_pipes.distributed_state_network.util")
    dsn_util.stop_thread = lambda thread: None
    sys.modules.setdefault("language_pipes.distributed_state_network.util", dsn_util)

    util_config = types.ModuleType("language_pipes.util.config")
    util_config.default_model_dir = lambda: "/tmp/models"
    util_config.default_config_dir = lambda: "/tmp/config"
    sys.modules.setdefault("language_pipes.util.config", util_config)


install_stub_modules()

from language_pipes.tui.content_provider.model_provider import ModelProvider, ModelStatus


class _Model:
    def __init__(self, model_id, loaded):
        self.model_id = model_id
        self.loaded = loaded


class _EndModelWithoutLoaded:
    def __init__(self, model_id):
        self.model_id = model_id


class ModelProviderStatusTests(unittest.TestCase):
    def test_get_models_status_reflects_model_manager_state(self):
        model_manager = sys.modules["language_pipes.modeling.model_manager"].ModelManager()
        model_manager.pipes_hosted = {
            "running/model": ["pipe-1"],
            "starting/model": ["pipe-2"],
            "stopped/model": ["pipe-3"],
            "end/running": ["pipe-4"],
            "end/starting": ["pipe-5"],
        }
        model_manager.models = [
            _Model("running/model", True),
            _Model("starting/model", False),
        ]
        model_manager.end_models = [
            _Model("end/running", True),
            _Model("end/starting", False),
        ]

        provider = ModelProvider(model_manager)

        self.assertEqual(
            provider.get_models_status(),
            {
                "running/model": ModelStatus.Running,
                "starting/model": ModelStatus.Starting,
                "stopped/model": ModelStatus.Stopped,
                "end/running": ModelStatus.Running,
                "end/starting": ModelStatus.Starting,
            },
        )

    def test_get_models_status_treats_end_models_without_loaded_flag_as_running(self):
        model_manager = sys.modules["language_pipes.modeling.model_manager"].ModelManager()
        model_manager.end_models = [_EndModelWithoutLoaded("end/default")]

        provider = ModelProvider(model_manager)

        self.assertEqual(
            provider.get_models_status(),
            {"end/default": ModelStatus.Running},
        )


if __name__ == "__main__":
    unittest.main()
