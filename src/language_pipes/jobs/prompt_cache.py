"""Node-local store of KV state for prompt prefixes that a later request can reuse.

The unit of identity is a hash chain over fixed-size blocks of prompt tokens
(`chain`), rooted at a scope that binds the entry to one origin node, one API key
and one `prompt_cache_key`. Every link is keyed under a secret that never leaves
this process, so an ID observed on the wire can be compared but neither extended
nor derived from a guessed prompt.

The store itself holds `CacheEntry` objects keyed by chain value. Entries share
the job's key/value tensors rather than copying them - `DynamicLayer.update`
rebinds `self.keys` to a fresh `torch.cat` result instead of writing into the old
tensor, so a snapshot taken by reference is frozen at the token count it was taken
at. Layer types that genuinely mutate in place (the linear-attention conv and
recurrent states) are cloned; `test_prompt_cache_share.py` is what keeps that
distinction honest across a transformers upgrade.
"""

import copy
import hmac
import logging
import secrets
import threading
from dataclasses import dataclass
from hashlib import sha256
from time import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from transformers.cache_utils import DynamicCache

from language_pipes.util.utils import CHUNK_SIZE, release_memory

# Tokens per chain link. Cached lengths are always a multiple of this, which is
# also the granularity OpenAI reports `cached_tokens` at.
BLOCK_SIZE = 128

# Prefixes shorter than this are never cached: the bookkeeping and the retained
# memory are not worth the prefill they would save.
MIN_CACHE_TOKENS = 256

# Cache boundaries have to fall on prefill chunk boundaries, or a resumed job
# would start mid-chunk.
assert BLOCK_SIZE % CHUNK_SIZE == 0, "BLOCK_SIZE must be a multiple of CHUNK_SIZE"

_DOMAIN = b"lp-prompt-cache-v1"


def new_secret() -> bytes:
    """A fresh chain key. One per `PromptCache`, so restarting the network from
    the TUI rotates it without restarting the process."""
    return secrets.token_bytes(32)


def scope(secret: bytes, origin_node_id: str, api_key: str, prompt_cache_key: str) -> bytes:
    """Root of the chain: `h[0]`, the ID of the empty prefix in this scope.

    Each field is digested before concatenation so the boundaries between them
    are unambiguous. With plain concatenation `api_key="ops"` +
    `prompt_cache_key="-ro:x"` and `api_key="ops-ro"` + `prompt_cache_key=":x"`
    would produce the same scope, and one tenant would read another's entries.
    """
    return hmac.new(
        secret,
        _DOMAIN + b"|scope"
        + sha256(origin_node_id.encode()).digest()
        + sha256(api_key.encode()).digest()
        + sha256(prompt_cache_key.encode()).digest(),
        sha256,
    ).digest()


def _link(secret: bytes, prev: bytes, block: Sequence[int]) -> bytes:
    """Extend a chain by one block.

    Keyed, not a plain digest: the intermediate values are observable by every
    node a job passes through, and an unkeyed link would let a holder of `h[i]`
    test guesses for block `i+1` offline.
    """
    return hmac.new(
        secret,
        _DOMAIN + b"|blk" + prev + np.asarray(block, dtype="<i4").tobytes(),
        sha256,
    ).digest()


def chain(secret: bytes, scope_id: bytes, tokens: Sequence[int]) -> List[bytes]:
    """IDs for every whole-block prefix of `tokens`.

    Index `i` names the prefix that is exactly `i * BLOCK_SIZE` tokens long, so
    index 0 is the scope itself (the empty prefix) and is never stored.
    """
    ids = [scope_id]
    prev = scope_id
    for start in range(0, (len(tokens) // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE):
        prev = _link(secret, prev, tokens[start:start + BLOCK_SIZE])
        ids.append(prev)
    return ids


def cache_ram_gb(cache: DynamicCache) -> float:
    """Bytes held by a cache's key/value tensors, in GB. Reporting only."""
    total_bytes = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            # Linear-attention layers keep a dict of states per layer; the
            # attention layers keep a single tensor.
            tensors = list(value.values()) if isinstance(value, dict) else [value]
            for tensor in tensors:
                if tensor is not None:
                    total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**3)


def _is_in_place_layer(layer) -> bool:
    """True for layer types whose update writes into a fixed buffer.

    `LinearAttentionLayer.update_recurrent_state` does `.copy_()` into a
    static-address tensor, so a borrower would corrupt whatever it shares. It
    also keeps its per-state bookkeeping in dicts that `lazy_initialization`
    mutates, which a shallow copy would share just as dangerously.

    Recognized by the presence of those attributes rather than by class name, so
    a new linear-attention variant is copied rather than silently shared.
    """
    return hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")


def copy_cache(cache: DynamicCache) -> DynamicCache:
    """A container that reads as `cache` does now and is immune to its future.

    Used by both `store` (freeze what the job has computed so far) and `adopt`
    (hand a job something it may append to). Growth-by-append layers are copied
    shallowly and so share their tensors, which is safe because appending
    rebinds the attribute on the layer object and every layer object here is a
    fresh one. Layers written in place are deep-copied instead; their states are
    fixed-size, so that does not scale with the prefix length.
    """
    layers = []
    for layer in getattr(cache, "layers", []):
        if _is_in_place_layer(layer):
            layers.append(copy.deepcopy(layer))
        else:
            layers.append(copy.copy(layer))

    # Shallow-copy the container so it keeps whatever the source was configured
    # with (layer class, offloading), then give it the fresh layer objects.
    copied = copy.copy(cache)
    copied.layers = layers
    return copied


@dataclass
class CacheEntry:
    cache_id: bytes
    # The only node allowed to reuse this entry. `cache_id` rides the packet on
    # a distributed pipe, so without this binding an observed ID would be a
    # bearer token that any peer could replay to a node it never shared a pipe
    # with.
    origin_node_id: str
    model_id: str
    # Every model process whose layers are in `cache`; a reload of any of them
    # invalidates the entry.
    process_ids: List[str]
    start_layer: int
    end_layer: int
    # i * BLOCK_SIZE, and this entry's charge against the budget.
    token_count: int
    cache: DynamicCache
    size_gb: float
    created: float
    last_used: float
    expires_at: float


@dataclass
class CacheStats:
    entries: int = 0
    tokens: int = 0
    reserved: int = 0
    budget: int = 0
    size_gb: float = 0.0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    no_store: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class _Reservation:
    job_id: str
    tokens: int


class PromptCache:
    """One per node, owned by `ContentProvider` next to the `JobTracker`.

    Every method is a no-op returning "miss" / `False` when either configured
    limit is 0, so a node with caching switched off does no work and holds no
    memory.
    """

    def __init__(
        self,
        get_max_cache_time: Callable[[], int],
        get_max_cache_tokens: Callable[[], int]
    ):
        self.get_max_cache_time = get_max_cache_time
        self.get_max_cache_tokens = get_max_cache_tokens
        # Per instance, never sent over the network, never written to disk,
        # never logged. Entries do not survive a restart because their IDs
        # cannot be named again.
        self._secret = new_secret()
        self._entries: Dict[bytes, CacheEntry] = {}
        self._reservations: Dict[str, _Reservation] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.no_store = 0
        self.logger = logging.getLogger(__name__)

    # -- identity ---------------------------------------------------------

    def scope(self, origin_node_id: str, api_key: str, prompt_cache_key: str) -> bytes:
        return scope(self._secret, origin_node_id, api_key, prompt_cache_key)

    def chain(self, scope_id: bytes, tokens: Sequence[int]) -> List[bytes]:
        return chain(self._secret, scope_id, tokens)

    # -- configuration ----------------------------------------------------

    def enabled(self) -> bool:
        return self.get_max_cache_time() > 0 and self.get_max_cache_tokens() > 0

    def _ttl(self, requested: Optional[int]) -> int:
        """Retention is a request, not a contract: the node's limit always wins."""
        limit = self.get_max_cache_time()
        if requested is None:
            return limit
        return min(requested, limit)

    # -- read path --------------------------------------------------------

    def lookup(
        self,
        cache_id: bytes,
        origin_node_id: str,
        model_id: str,
        process_ids: List[str],
        start_layer: int,
        end_layer: int
    ) -> Optional[CacheEntry]:
        """One ID, no counters. Reuse refreshes the entry's lifetime."""
        if not self.enabled():
            return None
        with self._lock:
            entry = self._entries.get(cache_id)
            if entry is None:
                return None
            if entry.expires_at <= time():
                return None
            if (
                entry.origin_node_id != origin_node_id
                or entry.model_id != model_id
                or entry.process_ids != process_ids
                or entry.start_layer != start_layer
                or entry.end_layer != end_layer
            ):
                return None
            entry.last_used = time()
            entry.expires_at = entry.last_used + self._ttl(None)
            return entry

    def find_longest(
        self,
        cache_ids: List[bytes],
        max_blocks: int,
        origin_node_id: str,
        model_id: str,
        process_ids: List[str],
        start_layer: int,
        end_layer: int
    ) -> Optional[Tuple[int, CacheEntry]]:
        """Walk the chain down from `max_blocks` and take the first hit.

        Counts exactly one hit or one miss for the request, which is what makes
        the reported hit rate mean "requests that reused something".
        """
        if not self.enabled():
            return None
        max_blocks = min(max_blocks, len(cache_ids) - 1)
        for blocks in range(max_blocks, (MIN_CACHE_TOKENS // BLOCK_SIZE) - 1, -1):
            entry = self.lookup(
                cache_ids[blocks], origin_node_id, model_id,
                process_ids, start_layer, end_layer
            )
            if entry is not None:
                with self._lock:
                    self.hits += 1
                return blocks, entry
        with self._lock:
            self.misses += 1
        return None

    def adopt(self, entry: CacheEntry) -> DynamicCache:
        """A cache the borrowing job may append to, leaving the entry intact."""
        return copy_cache(entry.cache)

    # -- write path -------------------------------------------------------

    def store(
        self,
        cache_id: bytes,
        cache: DynamicCache,
        origin_node_id: str,
        model_id: str,
        process_ids: List[str],
        start_layer: int,
        end_layer: int,
        token_count: int,
        ttl: Optional[int] = None
    ) -> bool:
        """Freeze this node's slice at `token_count` tokens under `cache_id`.

        Costs no allocation: the snapshot shares the job's current tensors and
        the job's next append rebinds its own, not ours. What it costs is that
        those tensors stop being freed when the job grows past this boundary.
        """
        if not self.enabled() or token_count < MIN_CACHE_TOKENS:
            return False

        snapshot = copy_cache(cache)
        now = time()
        entry = CacheEntry(
            cache_id=cache_id,
            origin_node_id=origin_node_id,
            model_id=model_id,
            process_ids=list(process_ids),
            start_layer=start_layer,
            end_layer=end_layer,
            token_count=token_count,
            cache=snapshot,
            size_gb=cache_ram_gb(snapshot),
            created=now,
            last_used=now,
            expires_at=now + self._ttl(ttl)
        )
        with self._lock:
            self._entries[cache_id] = entry
        return True

    # -- budget -----------------------------------------------------------

    def used_tokens(self) -> int:
        with self._lock:
            return self._used_tokens_locked()

    def _used_tokens_locked(self) -> int:
        return (
            sum(e.token_count for e in self._entries.values())
            + sum(r.tokens for r in self._reservations.values())
        )

    def reserve(self, job_id: str, tokens: int) -> bool:
        """Promise room for what this job will end up asking the cache to hold.

        Evicts least-recently-used entries until the estimate fits. A refusal is
        not a rejection of the job - it runs uncached, its own KV bounded by the
        job limits as it always was.
        """
        if not self.enabled():
            return False

        budget = self.get_max_cache_tokens()
        # A job that cannot fit an empty cache will not fit any cache, so there
        # is nothing to be gained by throwing entries away for it.
        if tokens > budget:
            with self._lock:
                self.no_store += 1
            return False

        evicted = 0
        with self._lock:
            if job_id in self._reservations:
                return True
            while self._used_tokens_locked() + tokens > budget and len(self._entries) > 0:
                self._evict_lru_locked()
                evicted += 1
            if self._used_tokens_locked() + tokens > budget:
                self.no_store += 1
                fits = False
            else:
                self._reservations[job_id] = _Reservation(job_id, tokens)
                fits = True
        if evicted > 0:
            release_memory()
        return fits

    def release(self, job_id: str):
        with self._lock:
            self._reservations.pop(job_id, None)

    def evict_lru(self) -> bool:
        """Drop the entry nothing has wanted for the longest.

        Safe against a job that is mid-decode on it: adoption handed that job a
        container of its own, and refcounting keeps the tensors alive as long as
        it holds them.
        """
        with self._lock:
            dropped = self._evict_lru_locked()
        if dropped:
            release_memory()
        return dropped

    def _evict_lru_locked(self) -> bool:
        if len(self._entries) == 0:
            return False
        oldest = min(self._entries.values(), key=lambda e: e.last_used)
        del self._entries[oldest.cache_id]
        self.evictions += 1
        return True

    def sweep(self):
        """Expire entries past their TTL. Runs on the tracker's 10s cadence."""
        now = time()
        with self._lock:
            expired = [k for k, e in self._entries.items() if e.expires_at <= now]
            for key in expired:
                del self._entries[key]
        if len(expired) > 0:
            release_memory()

    # -- lifecycle --------------------------------------------------------

    def clear(self):
        with self._lock:
            had = len(self._entries) > 0
            self._entries.clear()
            self._reservations.clear()
        if had:
            release_memory()

    def clear_process(self, process_id: str):
        """Drop everything computed by a model process that has been unloaded."""
        with self._lock:
            stale = [k for k, e in self._entries.items() if process_id in e.process_ids]
            for key in stale:
                del self._entries[key]
        if len(stale) > 0:
            release_memory()

    # -- reporting --------------------------------------------------------

    def stats(self) -> CacheStats:
        with self._lock:
            entries = list(self._entries.values())
            return CacheStats(
                entries=len(entries),
                tokens=sum(e.token_count for e in entries),
                reserved=sum(r.tokens for r in self._reservations.values()),
                budget=self.get_max_cache_tokens(),
                size_gb=sum(e.size_gb for e in entries),
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
                no_store=self.no_store
            )
