import unittest
from types import SimpleNamespace
from unittest.mock import patch
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)

from language_pipes.llm_layer_collector.auto.auto_layer import getClass as get_layer_class
from language_pipes.llm_layer_collector.auto.auto_rms import getClass as get_rms_class
from language_pipes.llm_layer_collector.auto.auto_rotary import getClass as get_rotary_class
from language_pipes.llm_layer_collector.auto.static_auto_model import StaticAutoModel


class TestGptOssSupport(unittest.TestCase):
    def test_auto_layer_maps_gpt_oss_decoder(self):
        self.assertEqual(get_layer_class(GptOssConfig()), GptOssDecoderLayer)

    def test_auto_rms_maps_gpt_oss_norm(self):
        self.assertEqual(get_rms_class(GptOssConfig()), GptOssRMSNorm)

    def test_auto_rotary_maps_gpt_oss_rotary(self):
        self.assertEqual(get_rotary_class(GptOssConfig()), GptOssRotaryEmbedding)

    @patch("language_pipes.llm_layer_collector.auto.static_auto_model.create_causal_mask")
    @patch("language_pipes.llm_layer_collector.auto.static_auto_model.GptOssModel.compute_embedding")
    def test_static_auto_model_embedding_dispatches_gpt_oss_to_gpt_oss_impl(
        self,
        mock_compute_embedding,
        mock_create_causal_mask,
    ):
        config = GptOssConfig()
        mock_create_causal_mask.return_value = None

        state = StaticAutoModel.compute_embedding(
            prompt_tokens=2,
            chunk_size=8,
            input_embedder=torch.nn.Embedding(16, 8),
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            config=config,
            cache=DynamicCache(),
        )

        self.assertEqual(state.state.shape, (1, 2, 8))
        mock_compute_embedding.assert_called_once()

    @patch("language_pipes.llm_layer_collector.auto.static_auto_model.GptOssModel.compute_layer")
    def test_static_auto_model_layer_dispatches_gpt_oss_to_gpt_oss_impl(self, mock_compute_layer):
        expected = torch.zeros((1, 1, 1), dtype=torch.float16)
        mock_compute_layer.return_value = expected

        layer: Any = SimpleNamespace(config=SimpleNamespace(model_type="gpt_oss"))
        state: Any = SimpleNamespace()
        cache = DynamicCache()

        out = StaticAutoModel.compute_layer(layer, state, cache)
        self.assertTrue(torch.equal(out, expected))
        mock_compute_layer.assert_called_once_with(layer, state, cache)


if __name__ == "__main__":
    unittest.main()