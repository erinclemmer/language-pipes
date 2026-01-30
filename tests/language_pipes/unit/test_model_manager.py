import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.config import LpConfig, HostedModel
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.modeling.meta_model import MetaModel
from language_pipes.modeling.llm_meta_data import LlmMetadata
from language_pipes.pipes.meta_pipe import MetaPipe
from language_pipes.pipes.router_pipes import RouterPipes


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeConnection:
    def __init__(self, address: str):
        self.address = address


class FakeStateNetworkNode:
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._data = {}
        self._peers = []
        self.logger = None

    def node_id(self):
        return self._node_id

    def read_data(self, node_id: str, key: str):
        return self._data.get((node_id, key))

    def update_data(self, key: str, value: str):
        self._data[(self._node_id, key)] = value

    def peers(self):
        return self._peers

    def connection_from_node(self, node_id: str):
        return FakeConnection("127.0.0.1")

    def add_peer(self, peer_id: str, models=None):
        self._peers.append(peer_id)
        if models is None:
            models = []
        import json
        self._data[(peer_id, "models")] = json.dumps([m.to_json() for m in models])


def make_metadata():
    """Create a fake LlmMetadata for testing."""
    metadata = LlmMetadata()
    metadata.embed_size = 128 * 10**6
    metadata.head_size = 256 * 10**6
    metadata.avg_layer_size = 64 * 10**6
    metadata.embed_hash = "embed_hash"
    metadata.head_hash = "head_hash"
    metadata.layer_hashes = ["l0", "l1", "l2", "l3"]
    return metadata


def make_config(
    node_id="node-a",
    hosted_models: Optional[List[HostedModel]] = None,
    max_pipes=2,
    model_validation=False
):
    """Helper to create a LpConfig with sensible defaults."""
    if hosted_models is None:
        hosted_models = []
    return LpConfig(
        logging_level="INFO",
        app_dir=".",
        model_dir="./models",
        oai_port=None,
        node_id=node_id,
        hosted_models=hosted_models,
        max_pipes=max_pipes,
        model_validation=model_validation,
        prefill_chunk_size=6,
    )


class FakeLlmModel:
    """Mock LlmModel for testing without loading real models."""
    
    def __init__(
        self, 
        model_id: str, 
        node_id: str, 
        pipe_id: str, 
        device: str,
        model_dir: str = "./models",
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

    def print(self, logger):
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
    
    def __init__(self, model_dir: str, model_id: str, device: str):
        self.model_dir = model_dir
        self.model_id = model_id
        self.device = device
        self.loaded = False

    def load(self):
        self.loaded = True

    def clean_up(self):
        pass


class ModelManagerTests(unittest.TestCase):
    """Tests for ModelManager class."""

    def _create_fake_llm_model(self, model_dir, model_id, node_id, pipe_id, device):
        """Factory function for creating FakeLlmModel instances."""
        return FakeLlmModel(
            model_id=model_id,
            node_id=node_id,
            pipe_id=pipe_id,
            device=device,
            model_dir=model_dir
        )

    @patch('language_pipes.modeling.model_manager.LlmModel')
    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_init_with_no_hosted_models(self, mock_llm_model):
        """Test ModelManager initializes correctly with no hosted models."""
        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        self.assertEqual(len(manager.models), 0)
        self.assertEqual(len(manager.end_models), 0)
        self.assertEqual(len(manager.pipes_hosted), 0)

    @patch('language_pipes.modeling.model_manager.LlmModel')
    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_stop_clears_models(self, mock_llm_model):
        """Test stop() clears all models and end_models."""
        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        # Manually add some fake models
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        manager.models.append(fake_model)
        fake_end_model = FakeEndModel("./models", "model-1", "cpu")
        manager.end_models.append(fake_end_model)

        manager.stop()

        self.assertEqual(len(manager.models), 0)
        self.assertEqual(len(manager.end_models), 0)

    @patch('language_pipes.modeling.model_manager.LlmModel')
    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_get_end_model_returns_matching_model(self, mock_llm_model):
        """Test get_end_model returns the correct EndModel by model_id."""
        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        end_model_1 = FakeEndModel("./models", "model-1", "cpu")
        end_model_2 = FakeEndModel("./models", "model-2", "cpu")
        manager.end_models.append(end_model_1)
        manager.end_models.append(end_model_2)

        result = manager.get_end_model("model-2")

        self.assertIsNotNone(result)
        self.assertEqual(result.model_id, "model-2")

    @patch('language_pipes.modeling.model_manager.LlmModel')
    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    def test_get_end_model_returns_none_when_not_found(self, mock_llm_model):
        """Test get_end_model returns None when model_id is not found."""
        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        end_model = FakeEndModel("./models", "model-1", "cpu")
        manager.end_models.append(end_model)

        result = manager.get_end_model("nonexistent-model")

        self.assertIsNone(result)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_loads_end_model_when_load_ends_true(self, mock_llm_model_class):
        """Test _host_model loads EndModel when load_ends is True."""
        # Setup mock to return a fake model
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        mock_llm_model_class.from_id.return_value = fake_model
        
        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=1.0,  # 1 GB
            load_ends=True
        )
        config = make_config(hosted_models=[hosted_model])
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        # Should have loaded an end model
        self.assertEqual(len(manager.end_models), 1)
        self.assertEqual(manager.end_models[0].model_id, "model-1")
        self.assertTrue(manager.end_models[0].loaded)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_does_not_load_end_model_when_load_ends_false(self, mock_llm_model_class):
        """Test _host_model does not load EndModel when load_ends is False."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=1.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model])
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        self.assertEqual(len(manager.end_models), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_creates_new_pipe_when_none_exist(self, mock_llm_model_class):
        """Test _host_model creates a new pipe when no pipes exist for the model."""
        fake_model = FakeLlmModel("model-1", "node-a", "new-pipe", "cpu")
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,  # 10 GB - enough memory to load layers
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model], max_pipes=1)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        # Should have created at least one pipe
        self.assertGreater(len(manager.pipes_hosted), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_respects_max_pipes(self, mock_llm_model_class):
        """Test _host_model respects max_pipes configuration."""
        call_count = [0]
        
        def create_model(*args, **kwargs):
            call_count[0] += 1
            fake_model = FakeLlmModel("model-1", "node-a", f"pipe-{call_count[0]}", "cpu")
            fake_model.start_layer = 0
            fake_model.end_layer = 3
            return fake_model
        
        mock_llm_model_class.from_id.side_effect = create_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model], max_pipes=1)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        self.assertEqual(len(manager.pipes_hosted), 1)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_max_pipes_works_per_model_id(self, mock_llm_model_class):
        """Test _host_model respects max_pipes on a per model basis"""
        call_count = [0]
        
        def create_model(*args, **kwargs):
            call_count[0] += 1
            fake_model = FakeLlmModel("model-1", "node-a", f"pipe-{call_count[0]}", "cpu")
            fake_model.start_layer = 0
            fake_model.end_layer = 3
            return fake_model
        
        mock_llm_model_class.from_id.side_effect = create_model

        logger = FakeLogger()
        hosted_model_1 = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        hosted_model_2 = HostedModel(
            id="model-2",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model_1, hosted_model_2], max_pipes=1)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        self.assertEqual(len(manager.pipes_hosted), 2)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_host_model_adds_to_existing_pipe(self, mock_llm_model_class):
        """Test _host_model adds model segments to existing incomplete pipes."""
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

        fake_model = FakeLlmModel("model-1", "node-a", "existing-pipe", "cpu")
        fake_model.start_layer = 2
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model], max_pipes=2)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        node.add_peer("node-b", [existing_model])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        self.assertIn("existing-pipe", manager.pipes_hosted["model-1"])

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_get_model_for_pipe_returns_none_when_no_memory(self, mock_llm_model_class):
        """Test _get_model_for_pipe returns None when there's not enough memory."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        fake_model.meta_data.avg_layer_size = 500 * 10**6  # 500MB per layer
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        pipe = MetaPipe("pipe-1", "model-1", [])
        
        # Very small available memory
        available_memory = 10  # 10 bytes - not enough
        remaining_memory, model = manager._get_model_for_pipe("model-1", pipe, "cpu", available_memory)

        self.assertIsNone(model)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_get_model_for_pipe_calculates_layers_based_on_memory(self, mock_llm_model_class):
        """Test _get_model_for_pipe calculates correct number of layers based on available memory."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu", num_hidden_layers=10)
        fake_model.meta_data.avg_layer_size = 100 * 10**6  # 100MB per layer
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        config = make_config()
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        pipe = MetaPipe("pipe-1", "model-1", [])
        
        # Enough memory for ~5 layers (500MB available, 100MB per layer, -1 for buffer)
        available_memory = 500 * 10**6
        remaining_memory, model = manager._get_model_for_pipe("model-1", pipe, "cpu", available_memory)

        self.assertIsNotNone(model)
        # Should start at layer 0 for empty pipe
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 4)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    @patch('language_pipes.modeling.model_manager.validate_model')
    def test_get_model_for_pipe_validates_when_enabled(self, mock_validate, mock_llm_model_class):
        """Test _get_model_for_pipe validates model metadata when validation is enabled."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        mock_llm_model_class.from_id.return_value = fake_model
        mock_validate.return_value = False  # Validation fails

        logger = FakeLogger()
        config = make_config(model_validation=True)
        node = FakeStateNetworkNode("node-a")
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)
        
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
        remaining_memory, model = manager._get_model_for_pipe("model-1", pipe, "cpu", available_memory)

        # Should return None because validation failed
        self.assertIsNone(model)
        # Should have logged a warning
        warnings = [m for m in logger.messages if m[0] == "warning"]
        self.assertGreater(len(warnings), 0)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_models_are_loaded_after_init(self, mock_llm_model_class):
        """Test that models are loaded (load() is called) after initialization."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model], max_pipes=1)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        # All models in the manager should be loaded
        for model in manager.models:
            self.assertTrue(model.loaded)

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_models_added_to_network_router(self, mock_llm_model_class):
        """Test that hosted models are added to the network via router_pipes."""
        fake_model = FakeLlmModel("model-1", "node-a", "pipe-1", "cpu")
        fake_model.start_layer = 0
        fake_model.end_layer = 3
        mock_llm_model_class.from_id.return_value = fake_model

        logger = FakeLogger()
        hosted_model = HostedModel(
            id="model-1",
            device="cpu",
            max_memory=10.0,
            load_ends=False
        )
        config = make_config(hosted_models=[hosted_model], max_pipes=1)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        # Check that the model was added to the network
        import json
        models_data = node.read_data("node-a", "models")
        self.assertIsNotNone(models_data)
        models = json.loads(models_data)
        self.assertGreater(len(models), 0)


class ModelManagerMultipleModelsTests(unittest.TestCase):
    """Tests for ModelManager with multiple hosted models."""

    @patch('language_pipes.modeling.model_manager.EndModel', FakeEndModel)
    @patch('language_pipes.modeling.model_manager.LlmModel')
    def test_multiple_hosted_models(self, mock_llm_model_class):
        """Test ModelManager handles multiple hosted models correctly."""
        call_count = [0]
        
        def create_model(model_dir, model_id, node_id, pipe_id, device):
            call_count[0] += 1
            fake_model = FakeLlmModel(model_id, node_id, pipe_id, device)
            fake_model.start_layer = 0
            fake_model.end_layer = 3
            return fake_model
        
        mock_llm_model_class.from_id.side_effect = create_model

        logger = FakeLogger()
        hosted_models = [
            HostedModel(id="model-1", device="cpu", max_memory=5.0, load_ends=True),
            HostedModel(id="model-2", device="cpu", max_memory=5.0, load_ends=True),
        ]
        config = make_config(hosted_models=hosted_models, max_pipes=4)
        node = FakeStateNetworkNode("node-a")
        node.add_peer("node-a", [])
        router = RouterPipes(node)

        manager = ModelManager(logger, config, router)

        # Should have end models for both
        self.assertEqual(len(manager.end_models), 2)
        model_ids = [m.model_id for m in manager.end_models]
        self.assertIn("model-1", model_ids)
        self.assertIn("model-2", model_ids)


if __name__ == "__main__":
    unittest.main()
