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
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from hashlib import sha256
from queue import Empty, Queue
from time import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
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
        json.dumps({
            "domain": _DOMAIN,
            "node_id": origin_node_id,
            "api_key": api_key,
            "prompt_cache_key": prompt_cache_key
        }).encode('utf-8'),
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


# Every attribute a layer might hold tensors under. Linear-attention layers
# keep a dict of states per layer; the attention layers keep a single tensor.
_LAYER_TENSOR_ATTRS = ("keys", "values", "conv_states", "recurrent_states")


def cache_ram_gb(cache: DynamicCache) -> float:
    """Bytes held by a cache's key/value tensors, in GB. Reporting only."""
    total_bytes = 0
    for layer in getattr(cache, "layers", []):
        for name in _LAYER_TENSOR_ATTRS:
            value = getattr(layer, name, None)
            tensors = list(value.values()) if isinstance(value, dict) else [value]
            for tensor in tensors:
                if tensor is not None:
                    total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**3)


def layer_devices(cache: DynamicCache) -> List[Optional[torch.device]]:
    """One device per layer, read off whichever tensor the layer actually has.

    `None` for a layer with no tensors yet. This is what `CacheEntry` records
    at store time, so a later promotion can put a borrower's copy back where
    the layers expect it without asking the model where it lives.
    """
    devices = []
    for layer in getattr(cache, "layers", []):
        device = None
        for name in _LAYER_TENSOR_ATTRS:
            value = getattr(layer, name, None)
            tensors = list(value.values()) if isinstance(value, dict) else [value]
            for tensor in tensors:
                if tensor is not None:
                    device = tensor.device
                    break
            if device is not None:
                break
        devices.append(device)
    return devices


def _is_host_resident(devices: Sequence[Optional[torch.device]]) -> bool:
    """True when every layer is already on the host - a CPU-only node, where
    demotion has nothing to do and must not pretend otherwise."""
    return all(d is None or d.type == "cpu" for d in devices)


def move_cache(cache: DynamicCache, devices: Sequence[Optional[torch.device]]) -> None:
    """Rebind every tensor of `cache` onto `devices`, one entry per layer.

    Walks the same attribute set as `cache_ram_gb`, dicts and `None`s included.
    `Tensor.to()` returns a new tensor rather than writing through, so this is
    invisible to anything holding a layer object copied out beforehand - the
    same argument that makes eviction safe. A tensor already on its target
    device is left alone, so demoting an entry twice allocates nothing.
    """
    for layer, device in zip(getattr(cache, "layers", []), devices, strict=True):
        if device is None:
            continue
        for name in _LAYER_TENSOR_ATTRS:
            value = getattr(layer, name, None)
            if value is None:
                continue
            if isinstance(value, dict):
                for key, tensor in list(value.items()):
                    if tensor is not None and tensor.device != device:
                        value[key] = tensor.to(device)
            elif value.device != device:
                setattr(layer, name, value.to(device))


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


def copy_cache(
    cache: DynamicCache,
    devices: Optional[Sequence[Optional[torch.device]]] = None
) -> DynamicCache:
    """A container that reads as `cache` does now and is immune to its future.

    Used by both `store` (freeze what the job has computed so far) and `adopt`
    (hand a job something it may append to). Growth-by-append layers are copied
    shallowly and so share their tensors, which is safe because appending
    rebinds the attribute on the layer object and every layer object here is a
    fresh one. Layers written in place are deep-copied instead; their states are
    fixed-size, so that does not scale with the prefix length.

    `devices`, if given, moves the *copy* onto those devices afterward - this is
    how `adopt` promotes a host-resident entry for a borrower without touching
    the entry itself, since every tensor moved here belongs to a layer object
    this function just created.
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
    if devices is not None:
        move_cache(copied, devices)
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
    # Per-layer device the tensors were computed on, recorded at store time so
    # `adopt` can put a borrower's copy back where the layers expect it without
    # asking the model where it lives. A node hosting two segments of one pipe
    # can have them on different GPUs, so this is per layer, not per entry.
    layer_devices: List[Optional[torch.device]] = field(default_factory=list)
    # False once the tensors have been moved to the host tier. Always True on a
    # CPU-only node, where there is nowhere else to put them.
    on_device: bool = True


@dataclass
class CacheStats:
    entries: int = 0
    # Device-tier totals. Before tiering existed every entry was
    # device-resident, so these are what these fields already measured.
    tokens: int = 0
    reserved: int = 0
    budget: int = 0
    size_gb: float = 0.0
    # Host-tier totals.
    host_tokens: int = 0
    host_budget: int = 0
    host_size_gb: float = 0.0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    no_store: int = 0
    demotions: int = 0
    promotions: int = 0

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
        get_max_cache_tokens: Callable[[], int],
        get_max_cache_host_tokens: Callable[[], int] = lambda: 0,
        start_worker: bool = False
    ):
        self.get_max_cache_time = get_max_cache_time
        self.get_max_cache_tokens = get_max_cache_tokens
        # Defaults to 0 (host tier off) so a caller that only knows about the
        # device budget - every test written before tiering existed - gets
        # exactly today's behavior: device pressure evicts rather than demotes.
        self.get_max_cache_host_tokens = get_max_cache_host_tokens
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
        self.demotions = 0
        self.promotions = 0
        self.logger = logging.getLogger(__name__)
        # Job completion enqueues here rather than copying tensors on the
        # job-processing thread; `drain_demotions` (or the worker below) does
        # the actual host-directed copy.
        self._demote_queue: "Queue[bytes]" = Queue()
        self.shutdown = False
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self._worker = threading.Thread(target=self._demote_worker_loop, daemon=True)
            self._worker.start()

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
    ) -> Tuple[int, CacheEntry] | None:
        """Walk the chain down from `max_blocks` and take the first hit"""
        if not self.enabled():
            return None
        max_blocks = min(max_blocks, len(cache_ids) - 1)
        for blocks in range(max_blocks, (MIN_CACHE_TOKENS // BLOCK_SIZE) - 1, -1):
            entry = self.lookup(
                cache_ids[blocks], origin_node_id, model_id,
                process_ids, start_layer, end_layer
            )
            if entry is not None:
                return blocks, entry
        return None

    def count_lookup(self, hit: bool):
        """Score one request's outcome.

        `lookup` deliberately counts nothing - the origin probes several IDs per
        request and scoring each would make the hit rate meaningless - so the
        one caller that does a single lookup per request, the layer node in
        `CachePolicy.adopt_for_node`, says so here. `find_longest` is the
        origin's equivalent.
        """
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1

    def adopt(self, entry: CacheEntry) -> DynamicCache:
        """A cache the borrowing job may append to, leaving the entry intact.

        A device-resident entry is shared as before - free. A host-resident
        one is promoted: the *copy* handed to the borrower is moved back onto
        the devices it was computed on, while the entry itself stays on the
        host. So an entry is device-resident for exactly the lifetime of the
        job that created it, and host-resident forever after.
        """
        if entry.on_device:
            return copy_cache(entry.cache)
        promoted = copy_cache(entry.cache, devices=entry.layer_devices)
        with self._lock:
            self.promotions += 1
        return promoted

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
            layer_devices=layer_devices(snapshot),
            on_device=True,
            created=now,
            last_used=now,
            expires_at=now + self._ttl(ttl)
        )
        with self._lock:
            self._entries[cache_id] = entry
        return True

    # -- budget -----------------------------------------------------------
    #
    # Two tiers, one LRU each. `max_cache_tokens` bounds device-resident
    # entries plus live reservations - the same thing it always bounded, back
    # when every entry was device-resident. `max_cache_host_tokens` bounds
    # entries that have been demoted off the device. Only the host tier
    # actually evicts: device pressure demotes into the host tier instead,
    # which is strictly better than evicting outright as long as there is
    # somewhere for the entry to go (§1 of the design doc explains why).

    def used_tokens(self) -> int:
        """Device-resident tokens plus live reservations - the field's exact
        meaning from before tiering existed."""
        with self._lock:
            return self._used_device_tokens_locked()

    def used_host_tokens(self) -> int:
        with self._lock:
            return self._used_host_tokens_locked()

    def _used_device_tokens_locked(self) -> int:
        return (
            sum(e.token_count for e in self._entries.values() if e.on_device)
            + sum(r.tokens for r in self._reservations.values())
        )

    def _used_host_tokens_locked(self) -> int:
        return sum(e.token_count for e in self._entries.values() if not e.on_device)

    def reserve(self, job_id: str, tokens: int) -> bool:
        """Promise room for what this job will end up asking the cache to hold.

        Demotes least-recently-used device-resident entries until the estimate
        fits, falling back to evicting one outright when it cannot be demoted
        (the host tier is disabled, or this is a CPU node with nothing to
        demote to). A refusal is not a rejection of the job - it runs uncached,
        its own KV bounded by the job limits as it always was.
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

        changed = 0
        with self._lock:
            if job_id in self._reservations:
                return True
            while self._used_device_tokens_locked() + tokens > budget and len(self._entries) > 0:
                if not self._demote_lru_locked():
                    break
                changed += 1
            if self._used_device_tokens_locked() + tokens > budget:
                self.no_store += 1
                fits = False
            else:
                self._reservations[job_id] = _Reservation(job_id, tokens)
                fits = True
            if changed > 0:
                self._evict_host_lru_while_over_budget_locked()
        if changed > 0:
            release_memory()
        return fits

    def release(self, job_id: str):
        with self._lock:
            self._reservations.pop(job_id, None)

    def evict_lru(self) -> bool:
        """Drop the entry nothing has wanted for the longest, regardless of
        which tier it is in.

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

    def _evict_host_lru_locked(self) -> bool:
        host_entries = [e for e in self._entries.values() if not e.on_device]
        if len(host_entries) == 0:
            return False
        oldest = min(host_entries, key=lambda e: e.last_used)
        del self._entries[oldest.cache_id]
        self.evictions += 1
        return True

    def _evict_host_lru_while_over_budget_locked(self):
        host_budget = self.get_max_cache_host_tokens()
        while self._used_host_tokens_locked() > host_budget and self._evict_host_lru_locked():
            pass

    def _demote_lru_locked(self) -> bool:
        """Free device budget by demoting the least-recently-used
        device-resident entry, or evicting it outright if it cannot be
        demoted. Assumes the caller holds `self._lock`."""
        device_entries = [e for e in self._entries.values() if e.on_device]
        if len(device_entries) == 0:
            return False
        oldest = min(device_entries, key=lambda e: e.last_used)
        if self._demote_locked(oldest):
            return True
        del self._entries[oldest.cache_id]
        self.evictions += 1
        return True

    def _demote_locked(self, entry: CacheEntry) -> bool:
        """Move one entry's tensors to the host in place. False when there is
        nothing to move (a CPU node) or nowhere to put it (host tier
        disabled) - both of which leave the entry exactly as it was. Assumes
        the caller holds `self._lock`."""
        if not entry.on_device:
            return False
        if _is_host_resident(entry.layer_devices) or self.get_max_cache_host_tokens() <= 0:
            return False
        move_cache(entry.cache, [torch.device("cpu")] * len(entry.layer_devices))
        entry.on_device = False
        self.demotions += 1
        return True

    def demote(self, cache_id: bytes) -> bool:
        """Move one entry to the host tier without dropping it - the whole
        point being that it survives for the next request.

        The actual tensor copy happens off the lock: a hold on `self._lock`
        would block every lookup and store on this node for as long as the
        D2H/H2D takes. So this builds the host-resident copy first, then
        retakes the lock and rebinds only if the entry is still there and
        still device-resident - if it was evicted, or already demoted by a
        racing caller, in the meantime, this is a no-op rather than a stale
        write.
        """
        with self._lock:
            entry = self._entries.get(cache_id)
            if entry is None or not entry.on_device:
                return False
            if _is_host_resident(entry.layer_devices) or self.get_max_cache_host_tokens() <= 0:
                return False
            source = entry.cache
            devices = entry.layer_devices

        host_copy = copy_cache(source, devices=[torch.device("cpu")] * len(devices))

        with self._lock:
            entry = self._entries.get(cache_id)
            if entry is None or not entry.on_device or entry.cache is not source:
                return False
            entry.cache = host_copy
            entry.on_device = False
            self.demotions += 1
            return True

    def demote_for_job(self, job) -> None:
        """Queue every entry a finished job stored or adopted for demotion.

        Enqueues only - actually moving the tensors happens off whatever
        thread called this (see `drain_demotions`), because this is called
        from job completion, and a D2H there would stall the next token of
        every other job on the node.
        """
        touched = getattr(job.caching, "touched_ids", None)
        if not touched:
            return
        for cache_id in touched:
            self._demote_queue.put(cache_id)

    def drain_demotions(self) -> int:
        """Process everything currently queued for demotion, synchronously.

        The worker thread built in `__init__` calls this; tests that build a
        `PromptCache` with `start_worker=False` call it directly so demotion
        stays deterministic under test.
        """
        pending = []
        while True:
            try:
                pending.append(self._demote_queue.get_nowait())
            except Empty:
                break
        changed = 0
        for cache_id in pending:
            if self.demote(cache_id):
                changed += 1
        if changed > 0:
            with self._lock:
                self._evict_host_lru_while_over_budget_locked()
            release_memory()
        return changed

    def _demote_worker_loop(self):
        while not self.shutdown:
            try:
                cache_id = self._demote_queue.get(timeout=0.5)
            except Empty:
                continue
            self._demote_queue.put(cache_id)
            self.drain_demotions()

    def sweep(self, live_ids: Optional[Set[bytes]] = None):
        """Expire entries past their TTL. Runs on the tracker's 10s cadence.

        A layer node is never told a job finished, so its jobs age out through
        the tracker's own stale timeout rather than a clean completion. When
        the tracker passes `live_ids` - the union of every pending job's
        `touched_ids` - this also demotes any device-resident entry none of
        them still holds, which is what makes that eventually happen out
        there, and cleans up after cancelled and timed-out jobs on the origin
        too.
        """
        now = time()
        with self._lock:
            expired = [k for k, e in self._entries.items() if e.expires_at <= now]
            for key in expired:
                del self._entries[key]
            stale_device_ids = []
            if live_ids is not None:
                stale_device_ids = [
                    k for k, e in self._entries.items()
                    if e.on_device and k not in live_ids
                ]
        if len(expired) > 0:
            release_memory()

        demoted = 0
        for cache_id in stale_device_ids:
            if self.demote(cache_id):
                demoted += 1
        if demoted > 0:
            with self._lock:
                self._evict_host_lru_while_over_budget_locked()
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
            device_entries = [e for e in entries if e.on_device]
            host_entries = [e for e in entries if not e.on_device]
            return CacheStats(
                entries=len(entries),
                tokens=sum(e.token_count for e in device_entries),
                reserved=sum(r.tokens for r in self._reservations.values()),
                budget=self.get_max_cache_tokens(),
                size_gb=sum(e.size_gb for e in device_entries),
                host_tokens=sum(e.token_count for e in host_entries),
                host_budget=self.get_max_cache_host_tokens(),
                host_size_gb=sum(e.size_gb for e in host_entries),
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
                no_store=self.no_store,
                demotions=self.demotions,
                promotions=self.promotions
            )
