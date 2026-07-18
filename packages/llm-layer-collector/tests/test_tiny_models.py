"""Tier 1 — tiny synthetic per-architecture parity tests (default CI suite).

One parameterized test per registered ``model_type`` (from ``specs.py``). Each
builds a random tiny checkpoint from the real HF config class, then verifies the
collector's forward path against the HF reference model's own ``last_hidden_state``
in **float32** (tight tolerances). A wiring bug that produces correct shapes but
garbled numbers — the entire reason the skill has a Debugging Playbook — fails
here loudly instead of shipping.

Adding a model = adding one ``TinyModelSpec`` to ``specs.py``. Nothing here changes.

    python -m unittest tests.llm_layer_collector.test_tiny_models
    python -m unittest tests.llm_layer_collector.test_tiny_models -k ministral
"""

import tempfile
import unittest

import torch
from transformers.cache_utils import DynamicCache

from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector import StaticAutoModel
from llm_layer_collector.load_layer import (
    get_shard_data,
    fuse_moe_expert_weights,
)
from llm_layer_collector.auto.auto_layer import AutoDecoderLayer

from .specs import TinyModelSpec, TINY_MODEL_SPECS
from .synthetic import build_tiny_checkpoint

# float32 tolerances — the whole point of the tiny suite is that these are tight.
COS_MIN = 0.9999
MAX_ABS_DIFF = 1e-4
SEQ_LEN = 8
CHUNK = 3


def _base_model(hf_model: torch.nn.Module) -> torch.nn.Module:
    """The inner transformer that returns ``last_hidden_state`` (before lm_head)."""
    return hf_model.model if hasattr(hf_model, "model") else hf_model


def _hf_last_hidden(base, input_ids: torch.Tensor, cache=None) -> torch.Tensor:
    with torch.no_grad():
        out = base(
            input_ids=input_ids,
            use_cache=cache is not None,
            past_key_values=cache,
        )
    return out.last_hidden_state


def _our_forward(collector, input_ids, cache, chunk_size,
                 layers, emb, norm, ple):
    """Run one ``compute_embedding`` slice + full layer stack + norm, returning the
    normed hidden state for whatever slice ``compute_embedding`` selected."""
    # prompt_tokens bounds the prefill window; the real prompt length is what the
    # collector slices against the cache.
    prompt_tokens = input_ids.shape[1]
    state = StaticAutoModel.compute_embedding(
        prompt_tokens, chunk_size, emb, input_ids, collector.config, cache,
        per_layer_embedder=ple,
    )
    for lyr in layers:
        state.state = StaticAutoModel.compute_layer(lyr, collector.config, state, cache)
    return state, norm(state.state)


class _SpecRunner:
    """Holds the per-spec fixtures so the four phases share one checkpoint."""

    def __init__(self, test: unittest.TestCase, spec: TinyModelSpec, model_dir: str):
        self.t = test
        self.spec = spec
        torch.manual_seed(1234)
        self.ck = build_tiny_checkpoint(spec, model_dir)
        self.config = self.ck.model.config.get_text_config()
        self.collector = LlmLayerCollector(
            self.ck.model_dir, self.ck.cache_file, dtype=torch.float32
        )
        self.vocab = self.config.vocab_size
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, self.vocab, (1, SEQ_LEN))

    # ---- Phase 1: collector loading + state-dict completeness ----
    def phase_loading(self):
        t, col = self.t, self.collector

        n_shard_keys = self._count_shard_keys()
        t.assertEqual(len(col.layer_files), n_shard_keys,
                      "cache did not capture every shard key")

        emb = col.load_input_embedding()
        norm = col.load_norm()
        head = col.load_head()
        layers = col.load_layer_set(0, col.num_layers - 1)

        t.assertEqual(len(layers), self.config.num_hidden_layers)
        t.assertEqual(emb.weight.shape,
                      (self.config.vocab_size, self.config.hidden_size))
        t.assertEqual(norm.cls.weight.shape, (self.config.hidden_size,))
        t.assertEqual(head.weight.shape,
                      (self.config.vocab_size, self.config.hidden_size))

        self._assert_layer_state_dict_complete()
        return emb, norm, head, layers

    def _count_shard_keys(self) -> int:
        from safetensors import safe_open
        import os
        total = 0
        for f in os.listdir(self.ck.model_dir):
            if f.endswith(".safetensors"):
                total += len(list(safe_open(
                    os.path.join(self.ck.model_dir, f), framework="pt").keys()))
        return total

    def _assert_layer_state_dict_complete(self):
        """Replicate load_layer's state-dict assembly for layer 0 and assert the
        HF module accepted every key — catches the silent ``strict=False`` drop
        that produced uninitialized MoE experts / unscaled fp8 weights."""
        col = self.collector
        shard = get_shard_data(0, 0, torch.device("cpu"), col.model_dir,
                               col.layer_prefix, col.layer_files, torch.float32)
        prefix = f"{col.layer_prefix}0."
        layer_sd = {k[len(prefix):]: v for k, v in shard.items() if k.startswith(prefix)}
        self.t.assertGreater(len(layer_sd), 0, "no weights found for layer 0")

        torch.set_default_device("meta")
        module = AutoDecoderLayer(col.config, 0)
        torch.set_default_device("cpu")
        module = module.to_empty(device=torch.device("cpu"))

        fuse_moe_expert_weights(layer_sd, set(module.cls.state_dict().keys()))
        res = module.cls.load_state_dict(layer_sd, strict=False)

        self.t.assertEqual(list(res.unexpected_keys), [],
                           f"unexpected keys dropped for {self.spec.model_type}")
        # Missing keys are only tolerable if they are non-persistent buffers
        # (e.g. rotary inv_freq) that the module recomputes — never weights.
        bad_missing = [k for k in res.missing_keys
                       if k.endswith(".weight") or k.endswith(".bias")]
        self.t.assertEqual(bad_missing, [],
                           f"weights missing after load for {self.spec.model_type}")

    # ---- Phase 2: single-shot parity ----
    def phase_single_shot(self, emb, norm, layers):
        ple = self.collector.load_per_layer_embedder() if self.spec.ple else None
        base = _base_model(self.ck.model)
        ref = _hf_last_hidden(base, self.input_ids)

        cache = DynamicCache()
        state, ours = _our_forward(
            self.collector, self.input_ids, cache, SEQ_LEN + 4,
            layers, emb, norm, ple)

        self._assert_match(ref, ours, "single-shot")
        self._assert_quirks(state)
        return ref

    # ---- Phase 3: chunked-prefill parity ----
    def phase_chunked(self, emb, norm, layers, ref_single):
        ple = self.collector.load_per_layer_embedder() if self.spec.ple else None
        t = self.t
        prompt_tokens = self.input_ids.shape[1]
        cache = DynamicCache()
        num_chunks = (prompt_tokens + CHUNK - 1) // CHUNK
        last_normed = None
        for chunk_idx in range(num_chunks):
            state, normed = _our_forward(
                self.collector, self.input_ids, cache, CHUNK,
                layers, emb, norm, ple)
            expected_lo = chunk_idx * CHUNK
            expected_hi = min((chunk_idx + 1) * CHUNK - 1, prompt_tokens - 1)
            t.assertEqual(state.cache_position[0].item(), expected_lo)
            t.assertEqual(state.cache_position[-1].item(), expected_hi)
            t.assertEqual(cache.get_seq_length(),
                          min((chunk_idx + 1) * CHUNK, prompt_tokens))
            last_normed = normed
        t.assertEqual(cache.get_seq_length(), prompt_tokens)
        # Final chunk's last position must equal the single-shot final position.
        self._assert_match(ref_single[:, -1:, :], last_normed[:, -1:, :], "chunked")

    # ---- Phase 4: decode parity ----
    def phase_decode(self, emb, norm, layers):
        ple = self.collector.load_per_layer_embedder() if self.spec.ple else None
        t = self.t
        base = _base_model(self.ck.model)

        # HF: prefill the prompt, then one decode step for a fixed next token.
        hf_cache = DynamicCache()
        _hf_last_hidden(base, self.input_ids, hf_cache)
        next_tok = torch.randint(0, self.vocab, (1, 1))
        ref_decode = _hf_last_hidden(base, next_tok, hf_cache)

        # Ours: prefill (single chunk), then the remaining<=0 decode slice.
        prompt_tokens = self.input_ids.shape[1]
        cache = DynamicCache()
        _our_forward(self.collector, self.input_ids, cache,
                     prompt_tokens, layers, emb, norm, ple)
        full_ids = torch.cat([self.input_ids, next_tok], dim=1)
        state, ours = _our_forward(self.collector, full_ids, cache,
                                   prompt_tokens, layers, emb, norm, ple)

        t.assertEqual(tuple(state.state.shape),
                      (1, 1, self.config.hidden_size))
        self._assert_match(ref_decode, ours, "decode")

    # ---- shared assertions ----
    def _assert_match(self, ref: torch.Tensor, ours: torch.Tensor, label: str):
        ref_f = ref[:, -1, :].float()
        ours_f = ours[:, -1, :].float()
        cos = torch.nn.functional.cosine_similarity(ref_f, ours_f).item()
        mad = (ref_f - ours_f).abs().max().item()
        self.t.assertGreaterEqual(
            cos, COS_MIN, f"{self.spec.model_type} {label}: cosine {cos} < {COS_MIN}")
        self.t.assertLess(
            mad, MAX_ABS_DIFF, f"{self.spec.model_type} {label}: max_abs_diff {mad}")

    def _assert_quirks(self, state):
        t, spec = self.t, self.spec
        if spec.per_type_rope:
            for d, name in ((state.causal_mask, "causal_mask"),
                            (state.position_embeddings, "position_embeddings")):
                t.assertIn("full_attention", d, f"{name} missing full_attention")
                t.assertIn("sliding_attention", d, f"{name} missing sliding_attention")
        if spec.ple:
            pli = state.per_layer_inputs
            t.assertIsNotNone(pli)
            t.assertEqual(
                tuple(pli.shape),
                (1, SEQ_LEN, self.config.num_hidden_layers,
                 self.config.hidden_size_per_layer_input))
        if spec.fp8:
            self._assert_fp8_dequant()

    def _assert_fp8_dequant(self):
        """One dequantized layer-0 projection must exactly equal the reference
        model's (dequantized) weight — proves weight_scale_inv was applied."""
        col = self.collector
        shard = get_shard_data(0, 0, torch.device("cpu"), col.model_dir,
                               col.layer_prefix, col.layer_files, torch.float32)
        ref_params = dict(self.ck.model.named_parameters())
        matched = 0
        for key, tensor in shard.items():
            if not key.endswith("q_proj.weight"):
                continue
            # Map collector key back to the reference model's "model.*" name.
            ref_key = self._to_ref_key(key)
            if ref_key in ref_params:
                self.t.assertTrue(
                    torch.equal(tensor, ref_params[ref_key].detach().float()),
                    "fp8 dequantized weight != reference weight")
                matched += 1
                break
        self.t.assertGreater(matched, 0, "no fp8 projection weight matched")

    def _to_ref_key(self, collector_key: str) -> str:
        # Reverse the mistral3 multimodal renaming: language_model.model.* -> model.*
        if collector_key.startswith("language_model.model."):
            return "model." + collector_key[len("language_model.model."):]
        if collector_key.startswith("model.language_model."):
            return "model." + collector_key[len("model.language_model."):]
        return collector_key


def _run_spec(test: unittest.TestCase, spec: TinyModelSpec):
    with tempfile.TemporaryDirectory() as d:
        runner = _SpecRunner(test, spec, d)
        emb, norm, _head, layers = runner.phase_loading()
        ref_single = runner.phase_single_shot(emb, norm, layers)
        runner.phase_chunked(emb, norm, layers, ref_single)
        runner.phase_decode(emb, norm, layers)


class TestTinyModels(unittest.TestCase):
    pass


def _make_test(spec: TinyModelSpec):
    def test(self):
        _run_spec(self, spec)
    return test


for _spec in TINY_MODEL_SPECS:
    setattr(TestTinyModels, f"test_{_spec.model_type}", _make_test(_spec))


if __name__ == "__main__":
    unittest.main()
