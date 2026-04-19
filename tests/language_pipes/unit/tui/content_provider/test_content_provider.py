import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "src"))

pipe_manager_module = types.ModuleType("language_pipes.pipes.pipe_manager")
pipe_manager_module.PipeManager = type("PipeManager", (), {})
sys.modules[pipe_manager_module.__name__] = pipe_manager_module

router_pipes_module = types.ModuleType("language_pipes.pipes.router_pipes")
router_pipes_module.RouterPipes = type(
    "RouterPipes",
    (),
    {"__init__": lambda self, router: setattr(self, "router", router)},
)
sys.modules[router_pipes_module.__name__] = router_pipes_module

model_manager_module = types.ModuleType("language_pipes.modeling.model_manager")
model_manager_module.ModelManager = type("ModelManager", (), {})
sys.modules[model_manager_module.__name__] = model_manager_module

dsn_handler_module = types.ModuleType("language_pipes.distributed_state_network.handler")
dsn_handler_module.DSNodeServer = type("DSNodeServer", (), {})
sys.modules[dsn_handler_module.__name__] = dsn_handler_module

model_provider_module = types.ModuleType("language_pipes.tui.content_provider.model_provider")

class _FakeModelProvider:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.router_pipes = None

    def set_router_pipes(self, router_pipes):
        self.router_pipes = router_pipes

model_provider_module.ModelProvider = _FakeModelProvider
sys.modules[model_provider_module.__name__] = model_provider_module

network_provider_module = types.ModuleType("language_pipes.tui.content_provider.network_provider")
network_provider_module.NetworkProvider = type(
    "NetworkProvider",
    (),
    {"__init__": lambda self, get_router, set_router: None},
)
sys.modules[network_provider_module.__name__] = network_provider_module

psutil_module = types.ModuleType("psutil")
psutil_module.virtual_memory = lambda: SimpleNamespace(total=0, used=0)
sys.modules[psutil_module.__name__] = psutil_module

from language_pipes.content_provider.content_provider import ContentProvider

class ContentProviderTests(unittest.TestCase):
    @patch(
        "language_pipes.tui.content_provider.content_provider.psutil.virtual_memory",
        return_value=SimpleNamespace(total=34 * (1024**3), used=21 * (1024**3)),
    )
    def test_get_total_system_ram_returns_gigabytes(self, _virtual_memory):
        self.assertEqual(ContentProvider.get_total_system_ram(), 34.0)

    @patch(
        "language_pipes.tui.content_provider.content_provider.psutil.virtual_memory",
        return_value=SimpleNamespace(total=34 * (1024**3), used=21 * (1024**3)),
    )
    def test_get_used_system_ram_returns_gigabytes(self, _virtual_memory):
        self.assertEqual(ContentProvider.get_used_system_ram(), 21.0)
