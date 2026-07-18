"""Tier 2 — real-checkpoint smoke tests. Opt-in, hard-capped at 10 GB.

Gated behind ``LP_RUN_MODEL_TESTS`` so default CI never downloads:

    LP_RUN_MODEL_TESTS=1       python -m unittest tests.llm_layer_collector.test_real_models
    LP_RUN_MODEL_TESTS=1       ... test_real_models -k ministral
    LP_RUN_MODEL_TESTS=nightly ... test_real_models          # also runs short decode

Each model runs in a subprocess with an ``RLIMIT_AS`` ceiling (see ``memguard``),
and its peak RSS is asserted ``< 10 GB``. Models that can't fit in memory are
streamed one layer-window at a time via :func:`run_stack_windowed`, so even a
16 GB-resident checkpoint runs inside the cap. Expected values are derived from
``config.json`` / the safetensors index — never hardcoded.
"""

import os
import gc
import sys
import json
import unittest
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector import StaticAutoModel
from llm_layer_collector.auto.auto_layer import mapper as LAYER_MAPPER
from llm_layer_collector.state_obj import LLmComputationState

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from specs import RealModelSpec, REAL_MODEL_SPECS
from memguard import run_capped, MEMORY_BUDGET_BYTES

RUN = os.environ.get("LP_RUN_MODEL_TESTS", "")
NIGHTLY = RUN == "nightly"
LAYER_WINDOW = 10

# --------------------------------------------------------------------------- #
# Checkpoint location helpers (models live in the shared language_pipes cache)
# --------------------------------------------------------------------------- #
def _base_dir(model_id: str) -> Path:
    return Path.home() / ".cache" / "language_pipes" / "models" / model_id


def _model_dir(model_id: str) -> Path:
    return _base_dir(model_id) / "data"


def _cache_file(model_id: str) -> Path:
    return _base_dir(model_id) / "cache.json"


def ensure_model(model_id: str) -> None:
    base = _base_dir(model_id)
    assert os.path.exists(base), f"{model_id} does not exist"

# --------------------------------------------------------------------------- #
# Derived expected values (never hardcoded per model)
# --------------------------------------------------------------------------- #
def derive_num_keys(model_dir: Path) -> int:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return len(json.load(f)["weight_map"])
    from safetensors import safe_open
    total = 0
    for f in os.listdir(model_dir):
        if f.endswith(".safetensors"):
            total += len(list(safe_open(os.path.join(model_dir, f),
                                        framework="pt").keys()))
    return total

def compute_layers(collector: LlmLayerCollector, start: int, end: int, state: LLmComputationState, cache: DynamicCache):
    layers = collector.load_layer_set(start, end)
    with torch.inference_mode():
        for lyr in layers:
            state.state = StaticAutoModel.compute_layer(
                lyr, collector.config, state, cache).detach()
    lyr = None
    layers.clear()
    del layers
    gc.collect()
    torch.cuda.empty_cache()

def run_stack_windowed(
    collector: LlmLayerCollector,
    input_ids: torch.Tensor,
    emb: torch.nn.Embedding,
    ple: Optional[torch.nn.Module] = None
):
    """Full-prompt prefill, loading the decoder stack one memory-bounded window at
    a time. Returns ``(state, cache)`` with ``state.state`` = final hidden state."""
    prompt_tokens = input_ids.shape[1]
    cache = DynamicCache()
    state = StaticAutoModel.compute_embedding(
        prompt_tokens, prompt_tokens, emb, input_ids, collector.config, cache,
        per_layer_embedder=ple)

    n_layers = collector.config.num_hidden_layers
    start = 0
    while start < n_layers:
        end = min(start + LAYER_WINDOW, n_layers - 1)
        compute_layers(collector, start, end, state, cache)
        start = end + 1
    return state, cache


# --------------------------------------------------------------------------- #
# Subprocess workers (module-level so they are picklable for spawn)
# --------------------------------------------------------------------------- #
def _prepare(spec: RealModelSpec):
    """Shared setup: ensure the checkpoint, build the collector, tokenize."""
    ensure_model(spec.model_id)
    model_dir, cache_file = _model_dir(spec.model_id), _cache_file(spec.model_id)
    collector = LlmLayerCollector(model_dir, cache_file, dtype=torch.bfloat16)
    if spec.inject_rope_scaling and getattr(collector.config, "rope_scaling", 1) is None:
        # GLM-4.1V ships rope_scaling: null, which crashes at layer time (skill §6).
        collector.config.rope_scaling = {"rope_type": "default",
                                          "mrope_section": [16, 24, 24]}
    tok = AutoTokenizer.from_pretrained(
        model_dir, fix_mistral_regex="mistralai" in spec.model_id)  # type: ignore
    input_ids = tok(spec.prompt, return_tensors="pt")["input_ids"]  # type: ignore
    return collector, tok, input_ids, model_dir


def coherence_worker(spec: RealModelSpec) -> dict:
    """Prefill a factual prompt and greedily decode the next token. Returns a
    small, picklable summary; the peak RSS is captured by the memguard wrapper."""
    collector, tok, input_ids, model_dir = _prepare(spec)

    num_keys = derive_num_keys(model_dir)
    hidden = collector.config.hidden_size
    vocab = collector.config.vocab_size

    emb = collector.load_input_embedding()
    norm = collector.load_norm()
    head = collector.load_head()
    ple = collector.load_per_layer_embedder()

    assert len(collector.layer_files) == num_keys, "cache missed keys"
    assert tuple(emb.weight.shape) == (vocab, hidden)
    assert tuple(head.weight.shape) == (vocab, hidden)

    state, cache = run_stack_windowed(collector, input_ids, emb, ple)
    assert cache.get_seq_length() == input_ids.shape[1]

    token = StaticAutoModel.compute_head(head, norm(state.state), "cpu", temperature=0)
    decoded = tok.decode([token])  # type: ignore
    return {"decoded": decoded, "num_keys": num_keys,
            "seq_len": int(cache.get_seq_length())}


def chunked_prefill_worker(spec: RealModelSpec, chunk_size: int = 32) -> dict:
    """Chunked-prefill smoke: assert cache_position / get_seq_length bounds per
    chunk. Loads the decoder stack one memory-bounded window at a time."""
    collector, tok, input_ids, _ = _prepare(spec)
    prompt_tokens = input_ids.shape[1]
    emb = collector.load_input_embedding()
    ple = collector.load_per_layer_embedder()
    n_layers = collector.config.num_hidden_layers

    cache = DynamicCache()
    num_chunks = (prompt_tokens + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        state = StaticAutoModel.compute_embedding(
            prompt_tokens, chunk_size, emb, input_ids, collector.config, cache,
            per_layer_embedder=ple)
        assert state.cache_position[0].item() == chunk_idx * chunk_size
        assert state.cache_position[-1].item() == min(
            (chunk_idx + 1) * chunk_size - 1, prompt_tokens - 1)
        start = 0
        while start < n_layers:
            end = min(start + LAYER_WINDOW, n_layers - 1)
            compute_layers(collector, start, end, state, cache)
            start = end + 1
        assert cache.get_seq_length() == min(
            (chunk_idx + 1) * chunk_size, prompt_tokens)
    return {"seq_len": int(cache.get_seq_length()), "prompt_tokens": prompt_tokens}


def decode_worker(spec: RealModelSpec, num_tokens: int) -> dict:
    """Short greedy decode with per-token window reloads (nightly)."""
    collector, tok, input_ids, _ = _prepare(spec)
    emb = collector.load_input_embedding()
    norm = collector.load_norm()
    head = collector.load_head()
    ple = collector.load_per_layer_embedder()

    state, cache = run_stack_windowed(collector, input_ids, emb, ple)
    current = input_ids
    n_layers = collector.config.num_hidden_layers
    for _ in range(num_tokens):
        token = StaticAutoModel.compute_head(head, norm(state.state), "cpu", temperature=0)
        current = torch.cat([current, torch.tensor([[token]])], dim=1)
        state = StaticAutoModel.compute_embedding(
            current.shape[1], 1, emb, current, collector.config, cache,
            per_layer_embedder=ple)
        start = 0
        while start < n_layers:
            end = min(start + LAYER_WINDOW, n_layers - 1)
            compute_layers(collector, start, end, state, cache)
            start = end + 1
    generated = tok.decode(current[0, input_ids.shape[1]:])  # type: ignore
    return {"generated": generated, "num_tokens": num_tokens}

# --------------------------------------------------------------------------- #
# The test case
# --------------------------------------------------------------------------- #
@unittest.skipUnless(RUN, "set LP_RUN_MODEL_TESTS=1 (or =nightly) to run real-model tests")
class TestRealModels(unittest.TestCase):
    def _run_capped_ok(self, worker, spec, *args):
        result = run_capped(worker, spec, *args)
        self.assertTrue(result.ok, f"{spec.model_id} failed:\n{result.error}")
        self.assertLess(
            result.peak_rss_bytes, MEMORY_BUDGET_BYTES,
            f"{spec.model_id} peak RSS {result.peak_rss_gib:.2f} GiB exceeds cap")
        print(f"\n[{spec.model_id}] peak RSS = {result.peak_rss_gib:.2f} GiB")
        return result.value

    def _real(self, spec: RealModelSpec):
        if spec.model_type not in LAYER_MAPPER:
            self.skipTest(
                f"{spec.model_type} has no collector dispatch (auto_layer.mapper)")

        value = self._run_capped_ok(coherence_worker, spec)
        decoded = value["decoded"]
        print(f"[{spec.model_id}] greedy next token = {decoded!r}")
        if spec.expected_next is not None:
            self.assertIn(spec.expected_next.strip().lower(),
                          decoded.strip().lower(),
                          f"{spec.model_id} greedy continuation {decoded!r} "
                          f"missing {spec.expected_next!r}")

        self._run_capped_ok(chunked_prefill_worker, spec)

        if NIGHTLY:
            self._run_capped_ok(decode_worker, spec, 8)


def _make(spec: RealModelSpec):
    def test(self):
        self._real(spec)
    return test


for _spec in REAL_MODEL_SPECS:
    setattr(TestRealModels, f"test_{_spec.model_type}", _make(_spec))


if __name__ == "__main__":
    unittest.main()
