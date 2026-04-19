import os
import sys
import unittest
from unittest.mock import Mock
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.tui.components.models_hosted import ModelsHosted
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.model_provider import (
    ModelProvider,
    ModelToLoad,
)
from language_pipes.content_provider.provider_calls import ProviderCall


class _FakeLoader:
    def __init__(self):
        self.calls = []

    def call_provider(self, name, data=None):
        self.calls.append((name, data))
        if name == ProviderCall.validate_device_name:
            return True
        return None


class _FakeConfirm:
    def open(self, *args, **kwargs):
        return None


class ModelsHostedTests(unittest.TestCase):
    def test_add_model_saves_and_hosts(self):
        loader = _FakeLoader()
        hosted = ModelsHosted(loader, _FakeConfirm(), lambda: None, lambda: True)
        hosted.editing_model = True
        hosted.edit_model_id = "org/model"
        hosted.edit_device_name = "cpu"
        hosted.edit_device_memory = "12.5"
        hosted.edit_load_ends = True

        hosted.add_model()

        self.assertEqual(
            [call[0] for call in loader.calls],
            [
                ProviderCall.validate_device_name,
                ProviderCall.save_layer_models,
                ProviderCall.host_layer_model,
            ],
        )
        saved_models = loader.calls[1][1]
        self.assertEqual(len(saved_models), 1)
        self.assertEqual(saved_models[0].model_id, "org/model")
        self.assertEqual(loader.calls[2][1].model_id, "org/model")


class ModelProviderTests(unittest.TestCase):
    def test_host_model_delegates_to_model_manager(self):
        model_manager = Mock()
        provider = ModelProvider(model_manager)
        router_pipes = object()
        provider.set_router_pipes(router_pipes)

        model = ModelToLoad("org/model", False, "cpu", 8.0)
        provider.host_layer_model(model)

        model_manager.host_model.assert_called_once()
        args = model_manager.host_model.call_args.args
        self.assertIs(args[0], router_pipes)
        self.assertEqual(args[1], "org/model")
        self.assertEqual(args[2], 8.0)
        self.assertEqual(str(args[3]), "cpu")
        self.assertEqual(args[4], 0)


class _FakeRouterPipes:
    def __init__(self, router):
        self.router = router


class ContentProviderTests(unittest.TestCase):
    @patch(
        "language_pipes.tui.content_provider.content_provider.RouterPipes",
        _FakeRouterPipes,
    )
    def test_set_router_updates_model_provider_router_pipes(self):
        provider = ContentProvider()
        router = object()

        provider.set_router(router)

        self.assertIs(provider.router, router)
        self.assertIs(provider.router_pipes.router, router)
        self.assertIs(provider.model_provider.router_pipes, provider.router_pipes)
