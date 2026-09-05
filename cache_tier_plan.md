# Prompt Cache — Host Tiering (Phase 4a)

Design and phasing for one Phase 4 item, replacing two rows of `cache_phasing.md`
§6: **adopt-by-move** (§4.7) and **CPU demotion** (§4.2). They were written as
independent optimizations; combined they are one mechanism that is strictly
better than either.

Section references (§) point at `cache_plan.md` unless noted.

---

## 1. The idea

Adopt-by-move and CPU demotion are both answers to the same sentence in §4.7:

> from the first pass onward, memory holds `prefix` (entry) + `prefix + generated` (job).

Adopt-by-move deletes the entry to kill the duplicate. Demotion moves the entry
to host RAM instead. Same VRAM relief, and the entry survives for the next
request — which is the whole point of having a cache.

**The invariant:**

> An entry is device-resident only while some job is still using the tensors it
> shares. Once no job is, it lives on the host. Adoption promotes a copy onto the
> device for the borrower; the entry itself stays on the host.

Everything below follows from that line.

### What it costs and what it buys

Steady state on a GPU node, one 4096-token prefix, a 28-layer Qwen3-1.7B node at
~115 KB/token (`documentation/configuration.md`'s own figure) — so the prefix is
~460 MB per copy:

| | VRAM | Host RAM | Copy at adopt |
|---|---|---|---|
| Today | 460 MB (entry) + `prefix+generated` (job) | — | none |
| Adopt-by-move | `prefix+generated` (job) | — | none, but the entry is gone |
| **Host tiering** | `prefix+generated` (job) | 460 MB (entry) | one H2D, ~40-80 ms |

The H2D is the whole cost, and it is paid on every hit. Against it: prefilling
4096 tokens on that model is several hundred milliseconds of compute, so the hit
still wins by a wide margin — but the margin is narrower than it is today, and
this is the number to measure before making tiering the default (§8).

### Why it supersedes adopt-by-move

Adopt-by-move exists to drop `caching.prefix_len` from the admission estimate
(§4.7). Tiering drops the same term for the same reason — there is no second
device-resident copy — and does not destroy the entry to do it. Adopt-by-move
survives only as the last resort *below* tiering: when the host tier is also
full, hand the tensors over rather than refuse the job (step 6, optional).

---

## 2. Where residency is decided

Two moments create a device-resident copy of an entry's tensors, and both end
when a job ends:

| Moment | What happens now | Under tiering |
|---|---|---|
| `store` | The entry shares the storing job's current tensors — free (§ `PromptCache.store` docstring). | Unchanged. The entry is device-resident and costs nothing extra, because the job holds those tensors anyway. |
| `adopt` | `copy_cache` hands the borrower fresh layer objects sharing the same tensors — free. | If the entry is host-resident, the borrower's copy is moved to the device (H2D). If it is device-resident, unchanged. |
| Job completion | Reservation released. | Every entry this job stored or adopted is demoted, if nothing else holds it on device. |

So an entry is device-resident for exactly the lifetime of the job that created
it, and host-resident forever after. The first job pays nothing; every later one
pays an H2D.

**Demotion is always safe, with no refcount.** `Tensor.to()` returns a new
tensor rather than writing through, and `copy_cache` already gives every borrower
its own layer *objects*, so rebinding `entry.cache.layers[i].keys` is invisible
to anything that adopted it. This is the same argument that makes eviction safe
(§4.7), and it is why the refcount that `cache_phasing.md` §6 asked for is not
needed. Demoting early is a waste of a copy, never a correctness problem.
`test_prompt_cache_share.py` is where that invariant is already pinned down; the
new demote/promote assertions belong beside it.

### The layer-node delay

A layer node is never told a job finished — `jobs_completed` is consulted only on
the origin (`job_receiver.py:90`), and layer-node jobs age out through
`EXPIRED_JOB_TIME = 60` in the stale sweep. So completion-triggered demotion
fires up to 60 s late out there. Rather than add a protocol message, the 10 s
`sweep()` also demotes any device-resident entry with no live job holding it;
that covers layer nodes and cleans up after cancelled and timed-out jobs on the
origin too.

---

## 3. The budget

This is the one decision that is not mechanical. `max_cache_tokens` today bounds
"stored entries + reservations" with no notion of where they live, and §4.7's
whole argument for the estimate is a VRAM argument.

**Decision: `max_cache_tokens` becomes the *device* budget, and a second field
bounds the host tier.**

```
used_device = Σ token_count of device-resident entries + Σ reservations
used_host   = Σ token_count of host-resident entries
```

- `max_cache_tokens` — unchanged name, unchanged default (16384), sharpened
  meaning: device-resident tokens. Before tiering existed every entry was
  device-resident, so this is what the field already measured on a GPU node.
  Every sizing paragraph in `documentation/configuration.md` stays true.
- `max_cache_host_tokens` — new, defaults to `4 * max_cache_tokens` when absent
  from the TOML, so an upgraded node needs no config change. Host RAM is cheaper
  and more plentiful than VRAM; the multiplier is what turns that into cache
  capacity instead of just VRAM relief.

Consequences:

- **The host tier is the only one that evicts.** Device pressure demotes the LRU
  entry, which lands in the host tier; host pressure drops the LRU entry for
  real. A standard two-level LRU.
- **Admission never demotes.** A D2H of hundreds of megabytes on the admission
  path would put that latency in front of every new job. `reserve` keeps today's
  behavior exactly — evict device-tier LRU until the estimate fits — and relies
  on completion-time and sweep-time demotion to keep the device tier small. If
  measurement shows the device tier is routinely full at admission, revisit.
- **A CPU-only node has no tiers.** Where the entry's tensors are already on the
  host device, `demote` is a no-op and the entry stays counted in `used_device`,
  so `max_cache_tokens` keeps its exact current meaning and a CPU node does not
  silently acquire a 4x larger cache.

---

## 4. Data model

`jobs/prompt_cache.py`:

```python
@dataclass
class CacheEntry:
    ...
    # Per-layer device the tensors were computed on, recorded at demotion so
    # `adopt` can put a borrower's copy back where the layers expect it. A node
    # hosting two segments of one pipe can have them on different GPUs, so this
    # is per layer, not per entry.
    layer_devices: List[Optional[torch.device]]
    # False once the tensors have been moved to the host tier.
    on_device: bool
```

`layer_devices` is the only thing that makes promotion possible without asking
the model where its layers are, and it cannot go stale: a segment that moves to
another device is reloaded, a reload mints a new `process_id`, and `identity()`
already invalidates the entry on that (§ `CachePolicy.identity`).

New module-level helper beside `copy_cache`:

```python
def move_cache(cache, devices) -> None   # rebind every tensor in place, walking
                                          # the same attribute set as cache_ram_gb
                                          # (keys/values/conv_states/recurrent_states,
                                          # dicts and Nones included)
def copy_cache(cache, devices=None)      # existing; `devices` restores residency
                                          # for the copy it hands the borrower
```

`CacheStats` gains `host_tokens`, `host_size_gb`, `demotions`, `promotions`;
`tokens` / `size_gb` become the device tier so the existing TUI line keeps
meaning what it says.

`JobCache` gains `touched_ids: List[bytes]` — every entry this job adopted or
stored, which is what completion demotes.

---

## 5. Steps

Each is separately committable with tests green.

### 5.1 Residency bookkeeping (no behavior change)

`jobs/prompt_cache.py`. Add `layer_devices` / `on_device` to `CacheEntry`,
`move_cache`, and the `devices` argument to `copy_cache`. Add
`PromptCache.demote(cache_id)` / the promote path inside `adopt`, with no caller
yet. `store` records `layer_devices` from the snapshot it takes.

*Tests* — `test_prompt_cache_tiering.py` (new). Devices cannot be exercised on
CI, which has no GPU, so these use stub tensors in the style of
`test_prompt_cache_share.py`'s `cache_of(*layers)`: an object with `.device`,
`.to()`, `.numel()`, `.element_size()`. Assert that `move_cache` walks dicts and
skips `None`, that demoting rebinds the entry's own layer objects and leaves an
earlier `copy_cache` result untouched, that `layer_devices` survives a
demote/promote round trip on a two-device cache, and that a cache already on the
target device is left alone (no new tensors).

### 5.2 The two-tier budget

`jobs/prompt_cache.py`, `config.py`, `content_provider/job_provider.py`,
`content_provider/content_provider.py:121`, `tui/components/jobs_server/top_state.py`.

`used_device_tokens` / `used_host_tokens` split; `PromptCache.__init__` takes a
third callable `get_max_cache_host_tokens`; `_evict_lru_locked` becomes
`_demote_lru_locked` for the device tier and keeps evicting for the host tier;
`config.py` gains `max_cache_host_tokens` with the derived default; the TUI's
`{stats.size_gb:.1f} GB` line (`top_state.py:224`) becomes
`1.8 GB GPU / 4.1 GB host`.

*Tests* — `test_prompt_cache_budget.py` extended: device pressure demotes rather
than evicting; a demoted entry is still found by `lookup`; host pressure evicts
the host LRU; a host tier at 0 makes demotion an eviction (today's behavior); the
derived default; `CacheStats` splits correctly.

### 5.3 Demote at job completion and on sweep

`jobs/job_cache.py`, `jobs/cache_policy.py`, `jobs/job_tracker.py`.

- `JobCache.touched_ids`, appended in `CachePolicy.plan` and `adopt_for_node`
  (adopt) and in `store_tagged_pass` (store).
- `JobTracker.remove_job` captures the `Job` objects it filters out — it only has
  the id today — and calls `prompt_cache.demote_for_job(job)`. That covers
  `complete_job`, `cancel_job`, the `ABORT` path in `job_receiver.py:229`, and
  the stale sweep, which are all the ways a job ends.
- `PromptCache.sweep` also demotes device-resident entries the tracker no longer
  has a live job for. This is what covers layer nodes (§2).
- `release_memory()` after a demotion batch, as eviction already does.

**Do the copy off the calling thread.** `complete_job` runs on the job-processing
thread; a 460 MB D2H there stalls the next token of every other job on the node.
`demote_for_job` enqueues; a small worker thread drains. It must not hold
`self._lock` across the copy either — take the lock, mark the entry, copy
outside it, retake the lock and rebind only if the entry is still present and
still device-resident. Expose `drain_demotions()` and build the worker only when
asked (`start_worker=False` in tests) so the unit tests stay deterministic.

*Tests* — `test_prompt_cache_tiering.py`: completing a job demotes what it
touched and nothing else; a cancelled job demotes; the sweep demotes an entry no
live job holds; a job still running keeps its entry device-resident; the drain
is idempotent; an entry evicted between enqueue and drain is skipped.
`test_job_tracker.py`: `remove_job` reaches the demotion path.

### 5.4 Promote on adopt

`jobs/prompt_cache.py`, `jobs/cache_policy.py`. `adopt` restores
`entry.layer_devices` into the borrower's copy and counts a promotion. Both call
sites (`CachePolicy.plan`, `CachePolicy.adopt_for_node`) are unchanged — this is
entirely inside `adopt`.

*Tests* — the promoted copy is on the recorded devices, the entry is still on the
host afterwards, and `test_prompt_cache_share.py`'s append-does-not-corrupt-the-
entry assertions still hold when the borrower was promoted rather than shared.

### 5.5 Drop `prefix_len` from the estimate — the adopt-by-move win

`jobs/cache_policy.py`, `jobs/prompt_cache.py`.

`plan` reserves *before* it looks up today (`cache_policy.py:219-237`), so it has
to assume the largest possible prefix. To charge the real cost it has to look
first:

```
probe (no counting) → estimate → reserve → adopt, or forget
```

`estimate = (entry.token_count if entry is device-resident else 0)
            + prompt_tokens + max_completion_tokens`

`find_longest` gains `count=False`; the existing `count_lookup` (already there
for the layer-node path) scores the request after admission, so a hit followed by
a refused reservation is not counted as a hit — which is what happens today by
accident of the ordering.

On layer nodes, `cache_reserve_tokens` keeps its current wire meaning (the
origin's full estimate, prefix included) and each node subtracts its *own*
prefix when its own entry is host-resident. Residency differs per node, so this
has to be local; keeping the wire number unchanged also means a Phase 3 node
talking to a Phase 4 node merely over-reserves.

*Tests* — `test_prompt_cache_path.py`: a host-resident hit reserves
`prompt + completion`; a device-resident hit reserves the prefix too; a job that
fits only because of the reduction is admitted; a hit whose reservation is then
refused counts as neither hit nor miss and runs uncached.
`test_distributed_cache.py`: the layer node's own subtraction.

### 5.6 Adopt-by-move as the last resort (optional)

Only reachable now when the host tier is full *and* admission would otherwise
refuse: hand the tensors to the job and delete the entry. Same code as the
original §4.7 proposal, one branch deep in `reserve`, and the case is rare
enough that it is reasonable to skip it entirely and let the job run uncached.

---

## 6. Doc edits

| File | Change |
|---|---|
| `cache_plan.md` §4.2 | CPU demotion goes from "possible later" to the described mechanism; the "adoption copies nothing" paragraph gains the promote case. |
| `cache_plan.md` §4.7 | The duplicate-residency paragraph gets its resolution; the estimate formula gains the residency condition; adopt-by-move is demoted to the host-tier-full fallback. |
| `cache_phasing.md` §6 | The two rows collapse into one pointing here. |
| `documentation/configuration.md` | `max_cache_host_tokens`; `max_cache_tokens` reworded as the device budget; a note that the sizing arithmetic now describes VRAM specifically. |
| `documentation/architecture.md`, `documentation/privacy.md` | Entries may live in host RAM. Privacy is unchanged — same process, same lifetime, never written to disk — but the doc says "in memory only" and should say where. |
| `documentation/release-notes.md` | The new field and the behavior change. |

---

## 7. Open decisions

1. **Is the H2D worth it?** (§1) Everything here is a bet that host RAM plus a
   PCIe copy beats VRAM plus recompute. It is very likely true for long prefixes
   and clearly false for short ones. A floor — only demote entries above some
   token count — is a one-line addition, but the threshold should come from a
   measurement, not a guess.
2. **Default `max_cache_host_tokens` multiplier.** 4x is a guess. Zero would ship
   the mechanism switched off, which is the conservative way to land it.
3. **Pinned host memory** would roughly halve both transfers, at the cost of
   pinning hundreds of megabytes of unswappable host RAM per entry. Worth
   trying, not worth defaulting to.
4. **Longer TTL for host-resident entries.** They cost far less to keep, so the
   natural move is to let them live longer than `max_cache_time`. But that field
   is a documented retention contract in `documentation/privacy.md` and
   `oai.md`, so this should not happen by accident. Recommend: leave TTL alone.

---

## 8. What to measure before defaulting it on

- D2H and H2D wall time for a realistic prefix on the target hardware, against
  the prefill time it replaces. This decides item 1 above.
- Peak VRAM across a multi-turn chat, before and after — the number the whole
  change exists to move.
- `cached_tokens` hit rate before and after. It should go *up* (entries survive
  budget pressure that used to evict them); if it does not, the host tier is
  sized wrong.
- Time-to-first-token on a cache hit. This is the one that can get worse.
