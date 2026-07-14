"""Tier 0 — pure unit tests. No real checkpoint, KBs of memory, run every commit.

Covers the logic that has bitten before (per the skill's gotchas): MoE fusion,
fp8 dequant, cache/prefix derivation, the multimodal lm_head fallback, and the
compute_head sampling + compute_embedding chunk-slicing math. The old
``test_exceptions`` content lives here too — it never needed a real model, only a
constructed collector, which the synthetic checkpoint provides.
"""

import os
import json
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector import StaticAutoModel
from llm_layer_collector.cache import get_shard_files, build_cache_data
from llm_layer_collector.helpers import load_shard_tensor, get_config
from llm_layer_collector.load_layer import (
    files_to_load_for_layer,
    files_to_load_for_layers,
    fuse_moe_expert_weights,
    dequantize_fp8_weights,
)

from transformers.cache_utils import DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from .specs import (
    TinyModelSpec,
    KEY_STYLE_MISTRAL3_MM,
)
from .synthetic import build_tiny_checkpoint


# --------------------------------------------------------------------------- #
# cache.py
# --------------------------------------------------------------------------- #
class TestCache(unittest.TestCase):
    def test_get_shard_files_single(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "model.safetensors"), "w").close()
            # a decoy sharded file must be ignored when the single file exists
            open(os.path.join(d, "model-00001-of-00002.safetensors"), "w").close()
            self.assertEqual(get_shard_files(r"model-(\d+)-of-(\d+).safetensors", d),
                             ["model.safetensors"])

    def test_get_shard_files_multi_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("model-00003-of-00003.safetensors",
                         "model-00001-of-00003.safetensors",
                         "model-00002-of-00003.safetensors",
                         "not-a-shard.txt"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual(
                get_shard_files(r"model-(\d+)-of-(\d+).safetensors", d),
                ["model-00001-of-00003.safetensors",
                 "model-00002-of-00003.safetensors",
                 "model-00003-of-00003.safetensors"])

    def test_get_shard_files_empty_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                get_shard_files(r"model-(\d+)-of-(\d+).safetensors", d)

    def test_build_cache_data_key_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"a.weight": torch.zeros(2), "b.weight": torch.zeros(2)},
                      os.path.join(d, "model-00001-of-00002.safetensors"))
            save_file({"c.weight": torch.zeros(2)},
                      os.path.join(d, "model-00002-of-00002.safetensors"))
            mapping = build_cache_data(d, r"model-(\d+)-of-(\d+).safetensors",
                                       torch.device("cpu"))
            self.assertEqual(mapping, {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00001-of-00002.safetensors",
                "c.weight": "model-00002-of-00002.safetensors",
            })


# --------------------------------------------------------------------------- #
# load_layer.py pure functions
# --------------------------------------------------------------------------- #
class TestFilesToLoad(unittest.TestCase):
    def test_dedup_across_layers(self):
        cache = {
            "model.layers.0.a": "f1", "model.layers.0.b": "f1",
            "model.layers.1.a": "f2",
        }
        self.assertEqual(files_to_load_for_layer("model.layers.0.", cache), ["f1"])
        self.assertEqual(
            sorted(files_to_load_for_layers(0, 1, "model.layers.", cache)),
            ["f1", "f2"])

    def test_unknown_prefix_raises(self):
        with self.assertRaises(Exception):
            files_to_load_for_layer("model.layers.99.", {"model.layers.0.a": "f1"})

    def test_prefix_collision_trailing_dot(self):
        # layers.1. must NOT match layers.10. — the trailing dot is what saves us.
        cache = {"model.layers.1.a": "fA", "model.layers.10.b": "fB"}
        self.assertEqual(files_to_load_for_layer("model.layers.1.", cache), ["fA"])
        self.assertEqual(files_to_load_for_layer("model.layers.10.", cache), ["fB"])


class TestFuseMoE(unittest.TestCase):
    def _experts(self, n, out, inp):
        sd = {}
        for i in range(n):
            sd[f"mlp.experts.{i}.gate_proj.weight"] = torch.full((out, inp), float(i))
            sd[f"mlp.experts.{i}.up_proj.weight"] = torch.full((out, inp), float(i) + 0.5)
            sd[f"mlp.experts.{i}.down_proj.weight"] = torch.full((inp, out), float(i) + 0.9)
        return sd

    def test_fuse_shapes_order_and_removal(self):
        sd = self._experts(3, 4, 2)
        model_keys = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        fuse_moe_expert_weights(sd, model_keys)

        self.assertIn("mlp.experts.gate_up_proj", sd)
        self.assertIn("mlp.experts.down_proj", sd)
        # per-expert keys removed
        self.assertFalse(any(".experts.0." in k for k in sd))

        gate_up = sd["mlp.experts.gate_up_proj"]
        self.assertEqual(tuple(gate_up.shape), (3, 8, 2))  # cat(gate[4], up[4]) along dim=1
        # expert-index order preserved: expert 2 gate value is 2.0, up value 2.5
        self.assertTrue(torch.equal(gate_up[2, :4, :], torch.full((4, 2), 2.0)))
        self.assertTrue(torch.equal(gate_up[2, 4:, :], torch.full((4, 2), 2.5)))
        self.assertEqual(tuple(sd["mlp.experts.down_proj"].shape), (3, 2, 4))

    def test_no_fusion_when_module_keeps_per_expert(self):
        sd = self._experts(2, 4, 2)
        before = set(sd.keys())
        fuse_moe_expert_weights(sd, set())  # module does not expect fused param
        self.assertEqual(set(sd.keys()), before)

    def test_non_moe_untouched(self):
        sd = {"self_attn.q_proj.weight": torch.ones(4, 4)}
        before = {k: v.clone() for k, v in sd.items()}
        fuse_moe_expert_weights(sd, {"mlp.experts.gate_up_proj"})
        self.assertEqual(set(sd.keys()), set(before.keys()))
        self.assertTrue(torch.equal(sd["self_attn.q_proj.weight"],
                                    before["self_attn.q_proj.weight"]))


class TestDequantizeFp8(unittest.TestCase):
    def test_scalar_scale(self):
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sd = {"proj.weight": w.clone(), "proj.weight_scale_inv": torch.tensor([2.0])}
        dequantize_fp8_weights(sd)
        self.assertNotIn("proj.weight_scale_inv", sd)
        self.assertTrue(torch.equal(sd["proj.weight"], w * 2.0))

    def test_blockwise_scale_repeat_interleave(self):
        w = torch.ones(4, 4)
        scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2 block grid over 4x4
        sd = {"proj.weight": w.clone(), "proj.weight_scale_inv": scale.clone()}
        dequantize_fp8_weights(sd)
        expected = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)  # 4x4
        self.assertTrue(torch.equal(sd["proj.weight"], expected))

    def test_blockwise_non_divisible_raises(self):
        sd = {"proj.weight": torch.ones(4, 4),
              "proj.weight_scale_inv": torch.ones(3, 3)}
        with self.assertRaises(RuntimeError):
            dequantize_fp8_weights(sd)

    def test_activation_scale_dropped(self):
        sd = {"proj.weight": torch.ones(2, 2), "proj.activation_scale": torch.ones(1)}
        dequantize_fp8_weights(sd)
        self.assertNotIn("proj.activation_scale", sd)
        self.assertIn("proj.weight", sd)


# --------------------------------------------------------------------------- #
# helpers.py
# --------------------------------------------------------------------------- #
class TestHelpers(unittest.TestCase):
    def test_load_shard_tensor_unknown_key(self):
        with self.assertRaises(ValueError):
            load_shard_tensor({}, "/nonexistent", "missing.weight",
                              torch.device("cpu"), torch.float32)

    def test_get_config_unwraps_mistral3(self):
        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
        from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
        tc = Ministral3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=8)
        with tempfile.TemporaryDirectory() as d:
            Mistral3Config(text_config=tc).save_pretrained(d)
            self.assertEqual(get_config(d).model_type, "ministral3")

    def test_get_config_unwraps_gemma4(self):
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4Config, Gemma4TextConfig)
        tc = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=1, head_dim=8,
                              hidden_size_per_layer_input=8, vocab_size_per_layer_input=64)
        with tempfile.TemporaryDirectory() as d:
            Gemma4Config(text_config=tc).save_pretrained(d)
            self.assertEqual(get_config(d).model_type, "gemma4_text")

    def test_get_config_unwraps_gemma3(self):
        from transformers.models.gemma3.configuration_gemma3 import (
            Gemma3Config, Gemma3TextConfig)
        tc = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=8)
        with tempfile.TemporaryDirectory() as d:
            # real gemma3 wrappers carry a top-level eos_token_id, which get_config
            # copies down into the text config.
            Gemma3Config(text_config=tc, eos_token_id=[1, 2]).save_pretrained(d)
            cfg = get_config(d)
            self.assertEqual(cfg.model_type, "gemma3_text")
            self.assertEqual(cfg.eos_token_id, [1, 2])


# --------------------------------------------------------------------------- #
# layer_collector.py init / derivation logic
# --------------------------------------------------------------------------- #
def _llama_kwargs(**over):
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              head_dim=16, max_position_embeddings=64, tie_word_embeddings=False)
    kw.update(over)
    return kw


class TestCollectorInit(unittest.TestCase):
    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                LlmLayerCollector(d, os.path.join(d, "cache.json"))

    def test_cache_rebuild_and_roundtrip(self):
        spec = TinyModelSpec("llama", LlamaConfig, _llama_kwargs())
        with tempfile.TemporaryDirectory() as d:
            ck = build_tiny_checkpoint(spec, d)
            self.assertFalse(os.path.exists(ck.cache_file))
            col = LlmLayerCollector(ck.model_dir, ck.cache_file, dtype=torch.float32)
            self.assertTrue(os.path.exists(ck.cache_file))
            with open(ck.cache_file) as f:
                cache = json.load(f)
            self.assertEqual(set(cache.keys()),
                             {"layer_files", "layer_prefix",
                              "input_embed_name", "norm_name"})
            self.assertEqual(cache["layer_prefix"], "model.layers.")
            self.assertEqual(cache["input_embed_name"], "model.embed_tokens.weight")
            self.assertEqual(cache["norm_name"], "model.norm.weight")
            # Second construction reads the cache without rebuilding.
            col2 = LlmLayerCollector(ck.model_dir, ck.cache_file, dtype=torch.float32)
            self.assertEqual(col2.layer_files, col.layer_files)

    def test_nested_prefix_vision_excluded_norm_derived(self):
        """Multimodal derivation: nested layer prefix, vision_tower keys ignored,
        norm name derived from the embedding name."""
        with tempfile.TemporaryDirectory() as d:
            LlamaConfig(**_llama_kwargs()).save_pretrained(d)
            save_file({
                # decoy vision key ordered first — must not set the layer prefix
                "model.vision_tower.encoder.layers.0.weight": torch.zeros(2),
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2),
                "model.language_model.embed_tokens.weight": torch.zeros(2),
                "model.language_model.norm.weight": torch.zeros(2),
            }, os.path.join(d, "model.safetensors"))
            col = LlmLayerCollector(d, os.path.join(d, "cache.json"),
                                    dtype=torch.float32)
            self.assertEqual(col.layer_prefix, "model.language_model.layers.")
            self.assertEqual(col.input_embedding_layer_name,
                             "model.language_model.embed_tokens.weight")
            self.assertEqual(col.norm_layer_name, "model.language_model.norm.weight")

    def test_multimodal_lm_head_fallback(self):
        """Untied multimodal head keyed as language_model.lm_head.weight must be
        resolved, not silently fall back to the embedding (Ministral3 §8 bug)."""
        spec = TinyModelSpec("ministral3", Ministral3Config,
                             dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=16,
                                  max_position_embeddings=64, sliding_window=None,
                                  tie_word_embeddings=False),
                             key_style=KEY_STYLE_MISTRAL3_MM, fp8=True)
        with tempfile.TemporaryDirectory() as d:
            ck = build_tiny_checkpoint(spec, d)
            col = LlmLayerCollector(ck.model_dir, ck.cache_file, dtype=torch.float32)
            self.assertEqual(col.lm_head_name, "language_model.lm_head.weight")
            self.assertIn(col.lm_head_name, col.layer_files)

    def test_tied_embedding_head_uses_embedding(self):
        spec = TinyModelSpec("llama", LlamaConfig,
                             _llama_kwargs(tie_word_embeddings=True))
        with tempfile.TemporaryDirectory() as d:
            ck = build_tiny_checkpoint(spec, d)
            col = LlmLayerCollector(ck.model_dir, ck.cache_file, dtype=torch.float32)
            self.assertNotIn(col.lm_head_name, col.layer_files)
            head = col.load_head()
            emb = col.load_input_embedding()
            self.assertTrue(torch.equal(head.weight, emb.weight))

    def test_load_in_8bit_forces_float16(self):
        spec = TinyModelSpec("llama", LlamaConfig, _llama_kwargs())
        with tempfile.TemporaryDirectory() as d:
            ck = build_tiny_checkpoint(spec, d)
            col = LlmLayerCollector(ck.model_dir, ck.cache_file,
                                    dtype=torch.bfloat16, load_in_8bit=True)
            self.assertEqual(col.dtype, torch.float16)


# --------------------------------------------------------------------------- #
# static_auto_model.compute_head sampling
# --------------------------------------------------------------------------- #
class TestComputeHeadSampling(unittest.TestCase):
    def _head_and_state(self):
        # Identity head: logits == state values == [0, 1, 2, 3]; argmax is token 3.
        head = torch.nn.Linear(4, 4, bias=False)
        head.weight = torch.nn.Parameter(torch.eye(4))
        state = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        return head, state

    def test_temperature_zero_is_greedy_argmax(self):
        head, state = self._head_and_state()
        self.assertEqual(
            StaticAutoModel.compute_head(head, state, "cpu", temperature=0), 3)

    def test_top_k_one_deterministic(self):
        head, state = self._head_and_state()
        for _ in range(5):
            self.assertEqual(
                StaticAutoModel.compute_head(head, state, "cpu", top_k=1,
                                             temperature=1), 3)

    def test_top_p_keeps_at_least_one(self):
        head, state = self._head_and_state()
        # Tiny top_p would remove everything but for the shift that keeps the top.
        self.assertEqual(
            StaticAutoModel.compute_head(head, state, "cpu", top_k=0,
                                         top_p=1e-6, temperature=1), 3)

    def test_min_p_threshold_filters(self):
        head, state = self._head_and_state()
        # min_p=0.9 keeps only tokens within 90% of the max prob → just token 3.
        self.assertEqual(
            StaticAutoModel.compute_head(head, state, "cpu", top_k=0, top_p=1,
                                         min_p=0.9, temperature=1), 3)

    def test_seeded_multinomial_reproducible(self):
        head, state = self._head_and_state()
        torch.manual_seed(42)
        a = StaticAutoModel.compute_head(head, state, "cpu", top_k=0, top_p=1,
                                         min_p=0, temperature=1)
        torch.manual_seed(42)
        b = StaticAutoModel.compute_head(head, state, "cpu", top_k=0, top_p=1,
                                         min_p=0, temperature=1)
        self.assertEqual(a, b)


# --------------------------------------------------------------------------- #
# static_auto_model.compute_embedding chunk-slicing math
# --------------------------------------------------------------------------- #
class TestComputeEmbeddingChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        spec = TinyModelSpec("llama", LlamaConfig, _llama_kwargs())
        ck = build_tiny_checkpoint(spec, cls._dir.name)
        cls.collector = LlmLayerCollector(ck.model_dir, ck.cache_file,
                                          dtype=torch.float32)
        cls.embedder = cls.collector.load_input_embedding()
        cls.config = cls.collector.config

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _advance(self, cache, n):
        """Populate the cache to seq_length n without running layers."""
        kv = torch.zeros(1, self.config.num_key_value_heads, n, self.config.head_dim)
        cache.update(kv, kv.clone(), 0)

    def test_first_chunk_window(self):
        cache = DynamicCache()
        ids = torch.randint(0, 128, (1, 8))
        state = StaticAutoModel.compute_embedding(8, 3, self.embedder, ids,
                                                  self.config, cache)
        self.assertEqual(state.cache_position[0].item(), 0)
        self.assertEqual(state.cache_position[-1].item(), 2)
        self.assertEqual(tuple(state.state.shape[:2]), (1, 3))
        self.assertEqual(tuple(state.position_ids.shape), (1, 3))

    def test_mid_prefill_window(self):
        cache = DynamicCache()
        self._advance(cache, 3)
        ids = torch.randint(0, 128, (1, 8))
        state = StaticAutoModel.compute_embedding(8, 3, self.embedder, ids,
                                                  self.config, cache)
        self.assertEqual(state.cache_position[0].item(), 3)
        self.assertEqual(state.cache_position[-1].item(), 5)

    def test_decode_slice_remaining_leq_zero(self):
        # Cache already holds the whole prompt; the next call must take exactly one
        # token (the appended one) — the "first decode slice is empty" regression.
        cache = DynamicCache()
        self._advance(cache, 8)
        ids = torch.randint(0, 128, (1, 9))  # prompt(8) + 1 appended token
        state = StaticAutoModel.compute_embedding(8, 3, self.embedder, ids,
                                                  self.config, cache)
        self.assertEqual(tuple(state.state.shape[:2]), (1, 1))
        self.assertEqual(state.cache_position[0].item(), 8)


if __name__ == "__main__":
    unittest.main()
