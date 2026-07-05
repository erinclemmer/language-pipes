import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Optional
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'language_pipes', 'unit'))

from language_pipes.config import LpConfig, ModelToLoad
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.modeling.meta_model import MetaModel
from language_pipes.modeling.llm_meta_data import LlmMetadata
from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import RouterPipes
from util import FakeLogger, FakeStateNetworkNode

import torch


def make_metadata():
    """Create a fake LlmMetadata for testing."""
    metadata = LlmMetadata()
    metadata.embed_size = 128 * 10**6
    metadata.head_size = 256 * 10**6
    metadata.avg_layer_size = 64 * 10**6
    metadata.embed_hash = "embed_hash"
    metadata.head_hash = "head_hash"
    metadata.layer_hash = "l0"
    metadata.version = "1.0.0"
    return metadata


def make_config(
    node_id="node-a",
    layer_models: Optional[List[ModelToLoad]] = None,
    end_models: Optional[List[str]] = None,
    num_local_layers: int = 0,
    max_pipes=2,
    model_validation=False
):
    """Helper to create a LpConfig with sensible defaults."""
    if layer_models is None:
        layer_models = []
    if end_models is None:
        end_models = []
    config = LpConfig()
    config.network_config.node_id = node_id
    config.layer_models = layer_models
    config.end_models = end_models
    return config


class FakeLlmModel:
    """Mock LlmModel for testing without loading real models."""
    
    def __init__(
        self, 
        model_id: str, 
        node_id: str, 
        pipe_id: str, 
        device: torch.device,
        model_dir: Path = Path("./models"),
        num_hidden_layers: int = 4
    ):
        self.model_id = model_id
        self.node_id = node_id
        self.pipe_id = pipe_id
        self.device = device
        self.model_dir = model_dir
        self.start_layer = -1
        self.end_layer = -1
        self.loaded = False
        self.num_hidden_layers = num_hidden_layers
        self.meta_data = make_metadata()
        self.process_id = f"process-{model_id}-{pipe_id}"
        
        # Mock collector with config
        self.collector = MagicMock()
        self.collector.config = MagicMock()
        self.collector.config.num_hidden_layers = num_hidden_layers

    def load(self):
        self.loaded = True

    def cleanup_tensors(self):
        pass

    def print(self, logger):  # pyright: ignore[reportGeneralIssue]
        logger.info(f"FakeLlmModel: {self.model_id} layers {self.start_layer}-{self.end_layer}")

    def to_meta(self) -> MetaModel:
        return MetaModel(
            process_id=self.process_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            node_id=self.node_id,
            pipe_id=self.pipe_id,
            model_id=self.model_id,
            loaded=self.loaded,
            num_layers=self.num_hidden_layers,
            meta_data=self.meta_data
        )


class FakeEndModel:
    """Mock EndModel for testing without loading real models."""
    
    def __init__(self, num_local_layers: int, model_dir: Path, model_id: str, device: str):
        self.model_dir = model_dir
        self.model_id = model_id
        self.device = device
        self.process_id = f"end-{model_id}"
        self.layers = list(range(num_local_layers))

    def load(self):
        pass

    def clean_up(self):
        pass


class ModelManagerTests(unittest.TestCase):
    """Tests for ModelManager class."""

    def test_init_with_no_layer_models(self):
        """Test ModelManager initializes correctly with no layer models."""
        manager = ModelManager()

        self.assertEqual(len(manager.layer_models), 0)
        self.assertEqual(len(manager.end_models), 0)
        self.assertEqual(len(manager.pipes_hosted), 0)

    def test_stop_clears_models(self):
        """Test stop() clears all models and end_models."""
        manager = ModelManager()
        # Manually add some fake models
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        manager.layer_models.append(fake_model)  # type: ignore[arg-type]
        fake_end_model = FakeEndModel(0, Path("models"), "model-1", "cpu")
        manager.end_models.append(fake_end_model)  # type: ignore[arg-type]

        manager.stop()

        self.assertEqual(len(manager.layer_models), 0)
        self.assertEqual(len(manager.end_models), 0)

    def test_get_end_model_returns_matching_model(self):
        """Test get_end_model returns the correct EndModel by model_id."""
        manager = ModelManager()
        end_model_1 = FakeEndModel(0, Path("./models"), "model-1", "cpu")
        end_model_2 = FakeEndModel(0, Path("./models"), "model-2", "cpu")
        manager.end_models.append(end_model_1)  # type: ignore[arg-type]
        manager.end_models.append(end_model_2)  # type: ignore[arg-type]

        result = manager.get_end_model("model-2")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.model_id, "model-2")

    def test_get_end_model_returns_none_when_not_found(self):
        """Test get_end_model returns None when model_id is not found."""
        manager = ModelManager()
        end_model = FakeEndModel(0, Path("./models"), "model-1", "cpu")
        manager.end_models.append(end_model)  # type: ignore[arg-type]

        result = manager.get_end_model("nonexistent-model")

        self.assertIsNone(result)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_end_model_loads(self):
        """Test load_end_model loads EndModel."""
        manager = ModelManager()

        manager.load_end_model("model-1", "cpu", 0)
        manager.load_end_model("model-2", "cpu", 0)

        self.assertEqual(len(manager.end_models), 2)
        self.assertEqual(manager.end_models[0].model_id, "model-1")
        self.assertEqual(len(manager.end_models[0].layers), 0)

        self.assertEqual(manager.end_models[1].model_id, "model-2")
        self.assertEqual(len(manager.end_models[1].layers), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_end_model_loads_layers(self):
        """Test load_end_model loads EndModel with layers."""
        manager = ModelManager()

        manager.load_end_model("model-1", "cpu", 2)
        manager.load_end_model("model-2", "cpu", 2)

        self.assertEqual(len(manager.end_models), 2)
        self.assertEqual(manager.end_models[0].model_id, "model-1")
        self.assertEqual(len(manager.end_models[0].layers), 2)

        self.assertEqual(manager.end_models[1].model_id, "model-2")
        self.assertEqual(len(manager.end_models[1].layers), 2)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_does_not_load_end_model(self, mock_llm_model_class):
        """Test host_model does not load EndModel."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        self.assertEqual(len(manager.layer_models), 1)
        self.assertEqual(manager.layer_models[0].start_layer, 0)
        self.assertEqual(len(manager.end_models), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_with_end_model_starts_after_end_layers(self, mock_llm_model_class):
        """Test host_model starts layer model after end model layers."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.load_end_model("model-1", "cpu", 1)
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=1, max_pipes=1)

        self.assertEqual(len(manager.layer_models), 1)
        self.assertEqual(manager.layer_models[0].start_layer, 1)
        self.assertEqual(len(manager.end_models), 1)
        self.assertEqual(len(manager.end_models[0].layers), 1)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_creates_new_pipe_when_none_exist(self, mock_llm_model_class):
        """Test host_model creates a new pipe when no pipes exist for the model."""
        fake_model = FakeLlmModel("model-1", "node-a", "new-pipe", torch.device("cpu"))
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        # Should have created at least one pipe
        self.assertGreater(len(manager.pipes_hosted), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_respects_max_pipes(self, mock_llm_model_class):
        """Test host_model respects max_pipes configuration."""
        call_count = [0]
        
        def create_model(*args, **kwargs):
            call_count[0] += 1
            fake_model = FakeLlmModel("model-1", "node-a", f"pipe-{call_count[0]}", torch.device("cpu"))
            fake_model.start_layer = 0
            fake_model.end_layer = 3
            return fake_model
        
        mock_llm_model_class.from_id.side_effect = create_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        self.assertEqual(len(manager.pipes_hosted), 1)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_max_pipes_works_per_model_id(self, mock_llm_model_class):
        """Test host_model respects max_pipes on a per model basis."""
        call_count = [0]
        
        def create_model(*args, **kwargs):
            call_count[0] += 1
            model_id = kwargs.get("model_id", "model-1")
            fake_model = FakeLlmModel(model_id, "node-a", f"pipe-{call_count[0]}", torch.device("cpu"))
            fake_model.start_layer = 0
            fake_model.end_layer = 3
            return fake_model
        
        mock_llm_model_class.from_id.side_effect = create_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)
        manager.host_model(router, "node-a", "model-2", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        self.assertEqual(len(manager.pipes_hosted), 2)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_adds_to_existing_pipe(self, mock_llm_model_class):
        """Test host_model adds model segments to existing incomplete pipes."""
        metadata = make_metadata()
        
        # Setup: Create an existing pipe with partial coverage
        existing_model = MetaModel(
            process_id="existing-process",
            start_layer=0,
            end_layer=1,
            loaded=True,
            node_id="node-b",
            pipe_id="existing-pipe",
            model_id="model-1",
            num_layers=4,
            meta_data=metadata
        )

        fake_model = FakeLlmModel("model-1", "node-a", "existing-pipe", torch.device("cpu"))
        fake_model.start_layer = 2
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        node.add_peer("node-b", [existing_model])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=2)

        self.assertIn("existing-pipe", manager.pipes_hosted["model-1"])

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_get_model_for_pipe_returns_none_when_no_memory(self, mock_llm_model_class):
        """Test _get_model_for_pipe returns None when there's not enough memory."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        fake_model.meta_data.avg_layer_size = 500 * 10**6  # 500MB per layer
        mock_llm_model_class.from_id.return_value = fake_model

        manager = ModelManager()
        pipe = MetaPipe("pipe-1", "model-1", [])
        
        # Very small available memory
        available_memory = 10  # 10 bytes - not enough
        remaining_memory, model = manager._get_model_for_pipe("node-a", "model-1", pipe, torch.device("cpu"), available_memory, 0)

        self.assertIsNone(model)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_get_model_for_pipe_calculates_layers_based_on_memory(self, mock_llm_model_class):
        """Test _get_model_for_pipe calculates correct number of layers based on available memory."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"), num_hidden_layers=10)
        fake_model.meta_data.avg_layer_size = 100 * 10**6  # 100MB per layer
        mock_llm_model_class.from_id.return_value = fake_model

        manager = ModelManager()
        pipe = MetaPipe("pipe-1", "model-1", [])
        
        # Enough memory for ~5 layers (500MB available, 100MB per layer, -1 for buffer)
        available_memory = 500 * 10**6
        remaining_memory, model = manager._get_model_for_pipe("node-a", "model-1", pipe, torch.device("cpu"), available_memory, 0)

        self.assertIsNotNone(model)
        assert model is not None
        # Should start at layer 0 for empty pipe
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 4)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_get_model_for_pipe_validates_when_enabled(self, mock_llm_model_class):
        """Test _get_model_for_pipe validates model metadata when validation is enabled."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        mock_llm_model_class.from_id.return_value = fake_model

        manager = ModelManager()
        
        # Create pipe with existing segment to trigger validation
        existing_segment = MetaModel(
            process_id="existing",
            start_layer=0,
            end_layer=1,
            loaded=True,
            node_id="node-b",
            pipe_id="pipe-1",
            model_id="model-1",
            num_layers=4,
            meta_data=make_metadata()
        )
        pipe = MetaPipe("pipe-1", "model-1", [existing_segment])
        
        available_memory = 1000 * 10**6
        remaining_memory, model = manager._get_model_for_pipe("node-a", "model-1", pipe, torch.device("cpu"), available_memory, 0)

        # With enough memory and matching metadata, model should be returned
        # (validate_model is no longer called inside _get_model_for_pipe)
        self.assertIsNotNone(model)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_models_are_loaded_after_host(self, mock_llm_model_class):
        """Test that models are loaded (load() is called) after host_model."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        # All models in the manager should be loaded
        for model in manager.layer_models:
            self.assertTrue(model.loaded)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_models_added_to_network_router(self, mock_llm_model_class):
        """Test that layer models are added to the network via router_pipes."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", torch.device("cpu"))
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager()
        manager.host_model(router, "node-a", "model-1", 10.0, torch.device("cpu"), first_layer=0, max_pipes=1)

        # Check that the model was added to the network
        models_data = node.read_data("node-a", "models")
        self.assertIsNotNone(models_data)
        assert models_data is not None
        models = json.loads(models_data)
        self.assertGreater(len(models), 0)

if __name__ == "__main__":
    unittest.main()
