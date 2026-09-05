# Prompt Caching — Phasing Plan

Companion to `cache_plan.md`. That document says *what* to build and *why*; this one
says *in what order*, *in which files*, and *what has to be true before each step
is done*. Section references (§) point at `cache_plan.md`.

Every phase is a separately reviewable, separately mergeable unit that leaves `main`
in a shippable state. Nothing in a later phase is needed to make an earlier phase
correct. Steps inside a phase are ordered so that each one can be committed with its
tests green before the next starts.

Conventions used below:

- File paths are relative to the repo root. Line numbers were as of commit `ab55c0e`
  and have been dropped from anything Phase 0 touched, since they have moved.
- A phase marked ✅ has shipped; its section records what was actually built, including
  where it departs from the plan as first written.
- "Tests green" means `pytest` (equivalently `python -m tests.language_pipes.unit`)
  passes without model weights, which is what CI runs. Integration tests under
  `tests/language_pipes/integration/` need weights and a GPU-less run is acceptable.
- Sizes are relative (S / M / L), not hours.

---

## 0. Where the code started

The plan was written against the tree at `ab55c0e`. A few facts drove the ordering and
are worth keeping as the baseline; the rows Phase 0 has since changed are marked:

| Fact | Where | Consequence for phasing |
|---|---|---|
| `Job.cache` is created in `Job.__init__` and dies with the job. | `src/language_pipes/jobs/job.py` | Every phase that retains KV has to hook `JobTracker.complete_job` / `remove_job` / the stale sweep. |
| Only the origin tokenizes; layer nodes create their `Job` in `JobTracker.add_job` from the first packet. | `src/language_pipes/jobs/job_tracker.py:127` | Layer-node adopt/reserve must live in or right beside `add_job`. |
| Prefill chunking is offset-free: `ChunkState.init(prompt_length)` starts at 0. | `src/language_pipes/util/chunk_state.py:17` | Resume-from-prefix needs an offset before any hit can be honored. |
| `StaticAutoModel.compute_embedding` already takes `past_seen_tokens` explicitly and slices `input_ids[:, past:past+take]`. | `packages/llm-layer-collector/src/llm_layer_collector/auto/static_auto_model.py:31-47` | No `llm_layer_collector` change is required for resume. |
| ~~A hash-validation failure bounces the packet to the origin via `restart_token`, and the origin then `advance()`s the chunk unconditionally.~~ | `jobs/job_receiver.py`, `jobs/job_processor.py` | **Fixed in Phase 0.** The origin now resends the saved pass; `_state_embed` is never re-entered on a bounce. |
| A job has at most one pass in flight: the origin waits for `HEAD` before dispatching the next. | `src/language_pipes/jobs/job_processor.py` | After a failure no node is more than one pass out of step, so replaying one saved output per node is a complete fix. Phase 0 relies on this; so does `attempt` in Phase 2. |
| `NetworkJob` serialization tolerates appended fields (`read_bytes` at EOF → `b''`, `read_int` → `0`), which is how `completed` / `progress` were added. | `src/language_pipes/jobs/network_job.py` | New wire fields are backward compatible if appended in order. Phase 0 added `pass_idx` last; Phase 2 appends after it. |
| Protocol dispatch is a flat `if` chain on an int: `0` job, `1` RFM, `CANCEL_PROTOCOL = 2`. | `src/language_pipes/content_provider/content_provider.py:150-157` | `CACHE_PROTOCOL = 3` slots in with one more branch. |
| Config limits are read fresh from TOML on every call (`JobProvider.get_max_node_jobs` does `LpConfig.from_file`). | `src/language_pipes/content_provider/job_provider.py:77-80` | The cache should take *callables* for its two limits, like `JobReceiver` takes `get_max_node_jobs`. |
| `do_POST` passes the literal `"anon"` as the API key when `api_keys` is empty. | `src/language_pipes/oai_server.py:50-53` | The §2.3 rule is decided at request-parse time and can be implemented entirely in `util/`. |
| Jobs / Server page has five focus rows: port, max node jobs, max api jobs, api keys, start/stop. | `src/language_pipes/tui/components/jobs_server/top_state.py:81-111` | Two rows insert at index 3 and 4; every `focus_idx` comparison after 2 shifts. |
| `EndModel` and `LlmModel` each carry a per-load `process_id` (uuid4). | `src/language_pipes/modeling/end_model.py:38`, `llm_model.py:84-87` | Entry invalidation on model reload needs no new plumbing; the id is already there. |
| Test fakes: `FakeEndModel`, `FakeModel`, `PipeWrapper`, `make_processor` in `tests/language_pipes/unit/util.py`. `make_processor` takes a `node_id` since Phase 0. | — | Cache plumbing must default to "off" when the processor is built without a cache, or every existing job-processor test breaks. |

---

## 1. Phase overview

| Phase | Deliverable | User-visible result | Depends on | Size |
|---|---|---|---|---|
| 0 ✅ | Job restart correctness (`pass_idx`, per-node saved output, replay instead of recompute) | A corrupted packet no longer silently desynchronizes the pipe's KV caches | — | M |
| 1 ✅ | Single-node prompt cache | `cached_tokens > 0` on the second of two prefix-sharing requests when the whole pipe is on the origin node; two config fields; TUI rows; docs | 0 | L |
| 2 ✅ | Distributed reuse | The same result across a multi-node pipe; `CacheStatus` protocol; per-node budgets | 0, 1 | L |
| 3 ✅ | Full OpenAI surface | `prompt_cache_options`, explicit breakpoints, `cache_write_tokens`, chat `stream_options.include_usage` | 1 (2 not required) | M |
| 4 | Optimizations | Adopt-by-move, CPU demotion, per-block snapshots, derived budget default | 2 | S each, independent |

Phase 3 depends only on Phase 1: every Phase 3 item is origin-side (parsing,
breakpoint→offset mapping, usage reporting). It can land before Phase 2 if the
protocol work runs long. Phase 4 items are individually optional.

---

## 2. Phase 0 — job restart correctness ✅

Landed in `f049727` (the fix) and `dfff22b` (the `PassSequence` extraction).

**Goal.** Make `restart_token` recovery correct with a *replay* of the failed pass,
and land the one wire field the cache protocol later reuses for drift detection
(`pass_idx`, §5.4). Reviewable without any cache context.

**Bug fixed** (was confirmed in code, §11):

- Prefill: the origin received the bounced packet with `compute_step = EMBED`,
  `data = None`. `_state_embed` saw `prompt_tokens != 0` and `chunking.is_active()`
  and called `chunking.advance()`, so the failed chunk was skipped, not retried.
- Decode: the origin re-embedded the current token; nodes upstream of the corruption
  point had already appended it and ended up with a duplicate KV position.

**Why replay works.** A job is strictly sequential: the origin dispatches pass N, waits
for it to return through `HEAD`, then dispatches N+1. When a hash fails, every node is
in exactly one of two states — it computed pass N or it did not — and nobody is more
than one pass out of step. So each node needs to keep only the output of the *last*
pass it computed at each point it was entered. On a retry, nodes that already computed
the pass resend that output without touching their cache; nodes that did not compute it
now. No cache is ever
rebuilt or cropped, no recompute happens, and the cost is one forwarding-only trip
through the nodes before the corruption point.

### What shipped

All of the bookkeeping lives in **`src/language_pipes/jobs/pass_sequence.py`** rather
than on `Job`, which carries only `Job.passes: PassSequence` and a three-line
`Job.replay(saved)` that applies a decision to its own fields. The module owns
`MAX_PASS_RETRIES = 3`, `PassKey`, `SavedPass` and the class:

| `PassSequence` member | Meaning |
|---|---|
| `idx` | The pass this node is handling. On the origin, the pass it dispatched. |
| `last_idx` | Highest pass number seen here; the sequence check reads it. |
| `key` | Entry point of the pass in hand, `(compute_step, current_layer)`. |
| `outputs` | `PassKey -> SavedPass`, what was forwarded for each entry point. |
| `retries` | Bounces served for the pass in flight. Origin only. |
| `replaying` | Set while forwarding a saved payload instead of computing. |
| `error` | Why a packet was refused, when the job cannot survive the refusal. |
| `start()` | Origin numbers a new pass (called from `_state_embed`). |
| `adopt(pass_idx)` | A layer node carries the origin's number. |
| `save(data, step, layer)` / `sent()` | Keep what went out; end a replay. Both from `_state_send`. |
| `restart(pass_idx)` | Answer a bounce → the pass to resend, or `None` to drop. |
| `accept(pass_idx, step, layer)` | → saved pass to forward, or `None` to compute; sets `error` to refuse. |

`PassSequence` decides and hands back a `SavedPass`; `Job` applies it. The tracker
never touches `Job` fields and `Job` never reasons about sequence numbers — keep that
split when Phase 2 adds `attempt`.

**0.1 — `pass_idx` on the wire.** `NetworkJob.pass_idx`, constructor default `0`,
written as the first appended field after `progress` and read in the same slot. The
Phase 2 fields append after it. The origin's own counter is `PassSequence.idx`,
incremented in `_state_embed`, which is where every pass starts.

**0.2 — Saved output per node.** `PassSequence.save()` is called from `_state_send`
with `job.data`, `job.compute_step` and `job.current_layer` — exactly the payload that
went out. It keeps the whole `JobData` (Gemma 4 mutates `shared_kv_states` as the pass
flows and the next node needs the post-mutation copy); `JobData` is already detached,
so holding it retains no graph. One chunk of hidden state per entry point, bounded by
`max_node_jobs` and freed with the job; `Job.get_job_ram` says in a comment that it
does not count it.

**0.3 — Replay-or-compute-or-refuse on receive.** `PassSequence.accept` is called from
`Job.receive_network_job` after the identity checks:

- Saved pass under this entry point with the same number → **replay**. `Job.replay`
  restores `data` / `compute_step` / `current_layer`; `replaying` sends the FSM
  straight to `SEND`, ahead of every other check in `get_next_state` and
  `_state_validating`. The cache is not touched.
- `pass_idx == 0` → peer that does not number passes; compute, check nothing.
- `last_idx == 0`, or `last_idx <= pass_idx <= last_idx + 1` → compute.
- Anything else → `error = "pass out of sequence"`. `Job.receive_network_job` returns
  `False` and `JobReceiver._process_network_job` cancels the job with that reason, so
  the origin's client gets an error instead of a stale timeout.

**0.4 — Origin handles the bounce by resending, not re-embedding.** A bounced packet is
distinguishable at the origin: `data is None` and `compute_step == EMBED`, which no
other packet has. `PassSequence.restart` answers it — same number (or `0`) as the pass
in flight → resend the payload saved under `(EMBED, 0)`, counting a retry; a lower
number is a second node reporting the same dead pass and is dropped. Because
`_state_embed` is never entered, the unconditional `advance()` is never reached; that
is the prefill fix, and the decode fix follows from 0.3 upstream. `restart_token` was
left alone apart from a docstring: it only edits `data`, `data_hash`, `compute_step`
and `current_layer`, so `pass_idx` already survives it. The fourth bounce for one pass
sets `error = "packet failed validation after 3 retries"` and cancels.

**0.5 — docs.** `documentation/job-processor.md` gained a "Restart" subsection under
`SEND` (with the decision table), a VALIDATING transition row and a diagram branch;
`documentation/architecture.md` "KV cache handling across nodes" now splits restart
from reroute and says a hash failure replays the pass and never rebuilds a cache.

### Decisions that differ from the plan as written

| Plan said | Shipped | Why |
|---|---|---|
| Passes number from `0`, `last_pass_idx = -1`, and a "constant `0` so far" flag detects an old peer. | Passes number from **1**, `last_idx = 0`, and `0` alone means "peer does not number passes". | With 0-based numbering the genuine first pass and "unnumbered" are the same value, so a bounce on the first pass would recompute — the exact bug being fixed. No flag needed. |
| One `last_pass_output` slot per job. | `outputs` keyed by the pass's **entry point**. | A node can host two layer ranges of one pipe and see the same pass twice (`TimingStats.record_completed_pass` already says so). One slot makes the second visit look like a replay of the first. |
| Save in `_state_process_layers`. | Save in `_state_send`. | An origin with no local end-model layers never enters `PROCESS_LAYERS` and would have nothing to replay. `_state_send` is exactly what went on the wire. |
| Replay calls `set_layer(saved.state, end_layer + 1, …)`, so the job needs the local `end_layer`. | Replay restores the saved `(data, compute_step, current_layer)` triple. | Same routing result, no layer-range lookup to plumb, and it also works when the saved step is `HEAD` (last node forwarding to the origin), which `set_layer` rejects. |
| — | A node whose `last_idx` is still `0` accepts any pass number. | Every layer node joins its job mid-stream via `JobTracker.add_job`; without this the first packet it ever sees is "out of sequence". |
| — | `JobReceiver._process_network_job` extracted from the runner loop body. | Makes the cancel-on-refusal path unit-testable. Behaviour is identical, including which exceptions reach the thread-restart handler. |

### Tests

| File | Covers |
|---|---|
| `test_network_job.py` | `pass_idx` round-trip; a payload truncated to the old writer's length parses with `pass_idx == 0`. |
| `test_job.py` — `JobReplayTests` | Replay resends the saved output, leaves the cache object and its `get_seq_length` alone, keeps the saved `shared_kv_states`; the next pass computes; a second visit at another entry point computes; out-of-sequence and already-passed refuse with the reason; a late joiner is accepted; a constant `0` always computes. |
| `test_job.py` — `JobRestartTests` | Bounce resends the pass in flight without changing its number; an unnumbered bounce is honored; a bounce for a dead pass is dropped; the retry cap; the counter resets with the next pass. |
| `job_processor/test_state_layers.py` | A repeated pass is forwarded without `process_job`, with the same `data` and `shared_kv_states` objects; the next pass runs the layers. |
| `job_processor/test_state_embed.py` | A bounced prefill chunk is resent with `compute_embed` never called and `chunking` unchanged; the next chunk still advances afterwards; a bounced decode token is resent with `current_token` and `input_ids` unchanged; a bounce for a replaced pass is dropped. |
| `job_processor/test_state_send.py` | The pass number goes on the wire; the payload is saved under the entry point; a replay ends when the pass goes back out. |
| `test_job_receiver.py` | Out-of-sequence cancels and notifies the origin; repeated validation failures cancel with the capped reason; a bounce for a dead pass is dropped. |
| `job_processor/test_restart.py` *(new file)* | Two real `Job`s across the real wire format with a real `restart_token`-shaped bounce, asserting KV positions on both nodes: a chunk lost outbound, a chunk lost on the return, the chunk that follows a restart, and a decode token. |

### Exit criteria

- ✅ 348 unit tests pass; the only existing tests touched were the ones being extended,
  plus a `node_id` parameter added to `make_processor` in `tests/…/unit/util.py`.
- ✅ `ruff` reports the same 60 pre-existing findings as before, none new.
- ⚠️ The hand-driven two-node run with a deliberately corrupted packet was **not** run:
  it needs model weights. `job_processor/test_restart.py` stands in for it and was
  falsified — all of its cases fail when the replay paths are disabled. Run the real
  two-node check before Phase 2 starts leaning on `pass_idx` for drift detection.
- ✅ Landed as its own commit, with no mention of caching in the code it touches.

---

## 3. Phase 1 — single-node prompt cache ✅

**Goal.** Everything in §3, §4.1, §4.2, §4.5 (implicit only), §4.6, §4.7, §2.3, §9
and the origin half of §4.3 / §4.4, with reuse gated to pipes whose every layer is
hosted on the origin. Ships a real, user-visible feature without any wire change.

The gate is deliberately conservative. A pipe is "local" when every segment
`pipe.get_layer(i)` for `i` in `[len(end_model.layers), num_hidden_layers)` has
`node_id == origin` and is not virtual. Add `Pipe.is_local_to(node_id) -> bool`
beside `Pipe.is_complete` (`src/language_pipes/pipes/pipe.py:67`). When it is
false, the request runs exactly as today; `cached_tokens` is 0.

The steps below landed as written apart from the departures recorded after them.
`Pipe.is_local_to` shipped as `is_local_to(node_id, start_layer)`, matching the
shape of `Pipe.is_complete(start_layer)` right above it, since the caller has to
say where the end model's own layers stop.

### Steps, in commit order

**1.1 — Config fields.** (S)

- `src/language_pipes/config.py`: `DEFAULT_MAX_CACHE_TIME = 300`,
  `DEFAULT_MAX_CACHE_TOKENS = 16384`; `LpConfig.max_cache_time` /
  `max_cache_tokens` next to `max_node_jobs` / `max_api_jobs` (lines 155-156, 167-168,
  180-181, 205-206, 255-256 all gain a twin).
- `JobProvider.get_/set_max_cache_time`, `get_/set_max_cache_tokens`
  (`job_provider.py`, same shape as lines 77-93).
- Tests: `test_config.py` — defaults when absent, round-trip through `save()` /
  `from_file`, `to_string()` lists them.
- Docs: `documentation/configuration.md` "API Server" — both fields, the "0 disables"
  rule, and the §3.1 sizing formula and worked Qwen3-1.7B example.

**1.2 — `jobs/prompt_cache.py`: chain and store.** (M)

- Module constants `BLOCK_SIZE = 128`, `MIN_CACHE_TOKENS = 256`, and an assertion
  that `BLOCK_SIZE % CHUNK_SIZE == 0` at import.
- Per-process secret via `secrets.token_bytes(32)`, created at module import or in
  `PromptCache.__init__` (one instance per node, either is per-process). Never
  logged, never serialized.
- `scope(origin_node_id, api_key, prompt_cache_key) -> bytes` and
  `chain(scope, tokens) -> List[bytes]` (§4.1, digested fields, keyed links).
- `CacheEntry` dataclass exactly as §4.2.
- `PromptCache(get_max_cache_time, get_max_cache_tokens)`:
  `lookup` (with origin / model / process / layer-range binding), `adopt`, `store`,
  `reserve`, `release`, `used_tokens`, `sweep`, `evict_lru`, `clear()` and
  `clear_process(process_id)`, plus the §9 counters. All methods are no-ops returning
  "miss" / `False` when either limit is 0.
- Locking: one `threading.Lock` around the entry table and the reservation table.
  `check_stale_jobs` runs on its own thread and the job runner on another.
- Memory release on evict/sweep: the same `gc.collect()` / `torch.cuda.empty_cache()`
  / `malloc_trim` sequence as `JobTracker.check_stale_jobs` — pull that into a small
  helper in `job_tracker.py` and call it from both places.
- Tests: `test_prompt_cache.py` and `test_prompt_cache_budget.py` as listed in §10,
  using a tiny `DynamicCache` built from a `PretrainedConfig` with two layers, no
  weights. The three threat-model tests (fresh secret differs, field framing, origin
  mismatch is a miss) go in here.

**1.3 — Sharing invariant test.** (S, but load-bearing)

- `test_prompt_cache_share.py`: adopt an entry, run one `DynamicLayer.update` and
  one `DynamicSlidingWindowLayer.update` on the adopted cache, assert the entry's
  tensors are unchanged in content and `data_ptr()`. For a
  `LinearAttentionLayer`-style state (or a stand-in that does `copy_()`), assert
  `adopt` cloned it. This test is what §4.2 says protects the design from a
  transformers upgrade. Land it with 1.2 so `adopt` is never merged without it.
- If any layer type fails the check, `adopt` clones that layer type and logs once.

**1.4 — Resume from an offset.** (S)

- `ChunkState.init(prompt_length, start_offset=0)`; `get_range` adds the offset;
  `is_active` / `has_more` / `is_final` derive from `prompt_length - start_offset`.
  `get_tokens_processed` keeps returning tokens covered by this job's own chunks.
- `JobCache.prefix_len: int = 0` (§1.5); `Job.past_seen_tokens()` returns
  `caching.prefix_len + chunking.get_tokens_processed()` during prefill; decode branch
  unchanged. `Job.init_chunking()` passes `caching.prefix_len` as the offset.
- Tests: extend `test_chunk_state.py` (offset ranges, offset + suffix shorter than
  `CHUNK_SIZE` stays inactive, `get_tokens_processed` excludes the offset) and
  `test_job.py` `JobPastSeenTokensTests` (prefill with prefix, first decode step
  after a prefix).

**1.5 — Request parsing and usage.** (M)

- New `src/language_pipes/util/oai_cache.py`:
  - `CacheOptions` dataclass: `prompt_cache_key: str`, `mode: str = "implicit"`,
    `ttl_seconds: Optional[int]`, `breakpoints: List[int]` (message indices; empty in
    Phase 1), `enabled: bool`.
  - `parse_cache_options(data, api_key, authenticated: bool) -> CacheOptions`. In
    Phase 1 it reads only `prompt_cache_key` (both endpoints) and applies §2.3:
    `enabled = authenticated or prompt_cache_key != ""`. `mode` / `ttl` /
    breakpoints are parsed in Phase 3; unknown-value 400s also wait for Phase 3 so
    that Phase 1 never rejects a request today's server accepts.
  - `usage_details(job)` helpers that emit `input_tokens_details.cached_tokens` /
    `prompt_tokens_details.cached_tokens` (`cache_write_tokens` is Phase 3). They read
    through `job.caching` defensively (`getattr`), because the usage block is built for
    whatever object the API layer was handed.
- `oai.py`: `ChatCompletionRequest` and `ResponsesRequest` gain `cache_options`;
  `from_dict` needs to know whether the server is authenticated, so thread
  `authenticated: bool` from `OAIHttpHandler` (`len(self.server.api_keys) > 0`)
  through `oai_chat_complete` / `oai_responses_create`. `_response_json` (line 253)
  and the chat usage block (line 409) and the streaming `response.completed` usage
  all call the helpers.
- New `src/language_pipes/jobs/job_cache.py`: `JobCache`, reached as `Job.caching`.
  It holds `options: CacheOptions`, `scope: bytes`, `ids: List[bytes]`,
  `prefix_len: int`, `cached_tokens: int`, `write_points: List[int]`,
  `pending_write_id: Optional[bytes]`, `pending_write_tokens: int` and
  `reserved: bool`, plus the decisions that need nothing else (§1.7). `Job` gains the
  single `caching` field; `Job.cache` keeps its old meaning, the working
  `DynamicCache` for the layers this node hosts.
- `JobFactory.start_job` accepts `cache_options` and stores it on `job.caching`; the
  `complete_cb` call sites in `oai.py` (line 418 and the responses equivalent) pass
  it. Keep the positional signature stable by adding it as a trailing keyword.
- Tests: `test_prompt_cache_anon.py` at the parse layer (§10), and
  `test_oai_responses.py` gains `usage.input_tokens_details` shape checks for
  streaming and non-streaming with `cached_tokens == 0` (the value is exercised in
  1.7).

**1.6 — Ownership and lifecycle.** (S)

- `ContentProvider.set_router` builds `PromptCache(job_provider.get_max_cache_time,
  job_provider.get_max_cache_tokens)` and hands it to `JobTracker(prompt_cache)`;
  `JobReceiver` reaches it via the tracker. `JobContext` gains
  `prompt_cache: Optional[PromptCache] = None`; `make_processor` in the test util
  leaves it `None`, and every cache branch in the processor is skipped when it is
  `None` or the job's `caching.options` is disabled.
- `JobTracker.complete_job` / `remove_job` / the stale sweep call
  `prompt_cache.release(job_id)`; `check_stale_jobs` calls `prompt_cache.sweep()` on
  its existing 10 s cadence.
- Drop entries on model unload: `ModelManager.shutdown_layer_models` /
  `shutdown_end_model` already collect the removed `process_id`s
  (`model_manager.py:196, 212`); add a hook alongside `set_job_hooks` that calls
  `prompt_cache.clear_process(process_id)`. `ContentProvider.stop_network` calls
  `clear()`.
- Tests: `test_job_tracker.py` — release on complete, cancel, and stale expiry;
  `test_model_manager.py` — unload clears entries for that process id.

**1.7 — Origin read/write path in the processor.** (M)

- `_state_embed`, in the `prompt_tokens == 0` branch after `tokenize`:
  1. If caching is off for the job, or the pipe is not local, or
     `prompt_tokens < MIN_CACHE_TOKENS`: skip.
  2. Compute `caching.scope` and `caching.ids` from the chain.
  3. Admission: `estimate = candidate_prefix + prompt_tokens + max_completion_tokens`;
     `reserve(job_id, estimate)`; on refusal, run uncached (log at info, one line,
     with the truncated scope hash).
  4. Walk `caching.ids` from the longest boundary `≤ prompt_tokens - 1` downward;
     first `lookup` hit → `adopt`, and `JobCache.adopt(blocks)` sets `prefix_len` and
     `cached_tokens`.
  5. `JobCache.plan_prompt_write(prompt_tokens)`: largest boundary `≤ prompt_tokens`,
     dropping any `≤ prefix_len` (already stored). The end-of-response point is
     decided at completion, not here.
  6. `init_chunking()` with the offset.
- Before each embed (both the tokenize branch and the `advance()` branch),
  `JobCache.tag(chunk_end)` sets `pending_write_id` / `pending_write_tokens` if the
  chunk about to be embedded ends on a write point (`JobCache.next_write_point`).
- `_state_process_layers`: after `end_model.compute_layers(job)` and after
  `model.process_job(job)`, `JobCache.take_pending()` claims the tag and clears it in
  one step; snapshot-validate (`job.data.cache_position[-1] + 1 == pending_write_tokens`)
  and `store`. A tag that is never claimed is cleared by the next `JobCache.tag()`
  instead, so it cannot survive into another pass (the origin is the only writer in
  Phase 1, so "cleared before the next pass" is trivially true).
- `_state_head`: when the job completes and mode is implicit,
  `JobCache.response_write_point(covered)` gives the largest boundary that is greater
  than every point already stored, or `None`; on a boundary, store the end-of-response
  entry. Note that at `HEAD` the cache covers
  `len(input_ids) - 1` positions (the last sampled token is never embedded), so the
  candidate is the largest boundary `≤ len(input_ids) - 1`. Release the reservation
  through the tracker as today.
- Tests: `job_processor/test_state_embed.py` — a local hit skips the cached chunks
  (count `compute_embed` calls); no hit when the pipe has a remote segment; write tag
  set on the chunk that ends on a boundary and only that chunk; admission refusal
  leaves `caching.prefix_len == 0` and no store. `test_state_layers.py` — store is
  called once with the tagged id and skipped when `cache_position` disagrees.
  `test_state_head.py` — end-of-response store at the right boundary.

**1.8 — TUI and stats.** (M)

- `top_state.py`: rows "Max Cache Time" (idx 3) and "Max Cache Tokens" (idx 4);
  api keys → 5, start/stop → 6. Update `_on_enter`, `_on_prev`, `_on_next`,
  `_get_tip_lines`, `get_footer`, the render block, and `_save_and_run` (persist both
  through `JobProvider`). Same digit-only editing as the two existing limit rows.
- `tips.py` `TIPS["jobs_server"]`: `max_cache_time`, `max_cache_tokens`, wording from
  §3.
- Stats line under the fields, from `JobProvider.get_cache_stats()` →
  `PromptCache.stats()`: entries, tokens used / budget, reserved, GB, hit rate.
- Active jobs view: "prefill skipped: N tokens" when `job.caching.cached_tokens > 0`
  (`MetaJob` gains `cached_tokens`).
- Tests: the `main_frame/components` suite has page tests for other pages; add one
  for the two new rows (navigation wraps at 6 / 5, enter on each row edits the right
  value).

**1.9 — Logging.** (S) Per-job completion line gains `cache=<hit|miss|off>
cached=<n> scope=<8 hex of h[0]>`. Never log `prompt_cache_key` or the API key.

**1.10 — Docs.** (S)

- `documentation/oai.md`: new "Prompt Caching" section — `prompt_cache_key` on both
  endpoints, the `cached_tokens` fields, 128-token granularity, 256-token minimum,
  the unauthenticated-server rule and the statement that `api_keys` is the supported
  isolation mechanism (§2.3). Note that `prompt_cache_options` and breakpoints arrive
  in a later release.
- `documentation/architecture.md`: rewrite the "KV cache handling across nodes"
  paragraph that says caches die with the job; in Phase 1 say reuse is limited to
  single-node pipes.
- `documentation/privacy.md`: a "Prompt cache retention" paragraph — derived KV state
  is held for up to `max_cache_time` after a request, in memory only, scoped by
  origin / API key / `prompt_cache_key`, and `max_cache_time = 0` restores the old
  behavior.
- `documentation/release-notes.md`: entry under the next release.

### Decisions that differ from the plan as written

| Plan said | Shipped | Why |
|---|---|---|
| `CacheEntry.process_id: str`. | `process_ids: List[str]`. | A Phase 1 entry covers the whole stack on the origin, which is the end model process *plus* every local layer segment's. `lookup` compares the list; `clear_process` drops any entry containing the id. Collapses to one element on a Phase 2 layer node. |
| `release_memory` helper in `job_tracker.py`. | In `util/utils.py`. | `JobTracker` now takes a `PromptCache`, so a helper in `job_tracker.py` that `prompt_cache.py` imports would be a cycle. |
| `lookup` is what the processor walks down. | `lookup` stays the single-id primitive; `find_longest` does the walk. | Counting hits and misses inside `lookup` would score one miss per candidate probed and make the reported hit rate meaningless. `find_longest` counts exactly one outcome per request. |
| Store after `end_model.compute_layers` *and* after `model.process_job`. | Store once, after `process_job`, when the pass has left `ComputeStep.LAYER`. | On a Phase 1 pipe the entry covers the whole stack, so a store after the end model's own layers would freeze a slice that is missing every later layer. Leaving `LAYER` is exactly the moment the last layer of the pass is done. |
| `parse_cache_options(data, api_key, authenticated)`. | `parse_cache_options(data, authenticated)`. | The API key is never needed to *parse*; the scope that uses it is derived in `JobFactory.start_job`, so the key travels no further than the call that already had it. |
| `adopt` clones `LinearAttention*Layer` states. | `copy_cache` deep-copies any layer carrying `conv_states` / `recurrent_states`. | In transformers 5.14 those are **dicts keyed by state index**, not bare tensors, and `lazy_initialization` mutates the sibling bookkeeping dicts too. A shallow copy would share all of it. Recognized by attribute rather than class name so a new variant is copied by default. |
| `reserve` evicts until the estimate fits. | An estimate larger than the whole budget refuses *before* evicting. | §4.7 asks for this: a job that cannot fit an empty cache will not fit any cache, so throwing entries away for it is pure loss. |
| — | `cache=off` in the log line covers "never looked", not just "opted out". | A short prompt, a remote segment and a refused admission are not cache misses, and scoring them as misses would understate the hit rate. Keyed off `JobCache.searched()`, i.e. the chain ids being empty. |
| Nine `cache_*` / `pending_write_*` fields directly on `Job`. | One `Job.caching: JobCache` (`jobs/job_cache.py`), holding the same state under shorter names plus the decisions that read only it. | The same split `PassSequence` got in Phase 0, for the same reason: `Job` is already a 380-line state machine, and the cache path is the one part of it a reader can take in on its own. It also gives Phase 2 somewhere to put the incoming `cache_use_*` tags without growing `Job` again. `Job.cache` (the working `DynamicCache`) deliberately keeps its name — it is KV state, not bookkeeping. |
| Seven `_cache_*` / `_plan_cache` / `_store_*` methods on `JobProcessor`. | One `CachePolicy` (`jobs/cache_policy.py`), built from the node's id, pipe, end model and `PromptCache`; the FSM holds one and calls into it at five points. | Same reason as `JobCache` and `PassSequence`: the cache block was 173 of the processor's 490 lines and touched no FSM state. The split is by question — the processor knows *when* a boundary is covered by what it just computed, the policy knows *whether* and *under which identity* to store it. It also gives Phase 2's layer-node paths (`JobTracker.add_job`, `JobReceiver`) somewhere to ask for an identity without a `JobContext`, instead of duplicating `_cache_identity`. `identity()` already takes the job for that reason, though Phase 1 ignores it. |
| — | Two new test files beyond the plan's list: `job_processor/test_prompt_cache_path.py` and `test_jobs_server_page.py`. | The plan named `main_frame/components` for the TUI page test; that directory does not exist in this tree, so the page test sits with the other top-level TUI tests. |

**Cross-cutting decisions (§7) as settled.** (1) Default **on**: `DEFAULT_MAX_CACHE_TIME
= 300`, `DEFAULT_MAX_CACHE_TOKENS = 16384`, which with the §2.3 anon rule only caches
for callers who authenticate or opt in with `prompt_cache_key`. (3) The secret is
**per `PromptCache` instance**, so restarting the network from the TUI rotates it
without restarting the process.

### Tests

| File | Covers |
|---|---|
| `test_prompt_cache.py` *(new)* | Chain determinism under one secret; divergence from the first differing block; scope separation by API key / cache key / origin; the three threat-model cases (fresh secret, field framing, origin mismatch); store/lookup binding by model, process ids and layer range; TTL expiry, refresh-on-use and clamping; `find_longest` taking the longest candidate and counting one outcome; both disable switches; `clear` / `clear_process`. |
| `test_prompt_cache_share.py` *(new)* | The sharing invariant, against real transformers layer objects: `DynamicLayer` and `DynamicSlidingWindowLayer` appends leave the entry's tensors unchanged in content and `data_ptr`; `cumulative_length` survives a snapshot; a `LinearAttentionLayer` entry is copied, its per-state dicts included, and a borrower's `copy_()` cannot reach it; a stored entry does not grow with the job. |
| `test_prompt_cache_budget.py` *(new)* | `used_tokens` counts entries plus reservations; LRU eviction at admission; an over-budget estimate evicts nothing; release on demand and on a double reserve; reuse moves an entry to the back of the order; eviction cannot break a job already running on it; the stats line. |
| `test_prompt_cache_anon.py` *(new)* | §2.3 at the parse layer, on both endpoints: no key on an open server disables caching, a key enables it, two keys do not share a scope, an absent key still caches when `api_keys` is set, and a non-string key is ignored rather than 400ed. |
| `job_processor/test_prompt_cache_path.py` *(new)* | The origin FSM: a hit starts the chunks after the prefix and skips them; a miss prefills everything; the whole prompt is never adopted; a remote segment, a disabled job, a short prompt and a refused admission all run uncached; write points and which chunk is tagged; store on the last layer of the pass and the drift check that skips it; the end-of-response entry; the two-request round trip; the log fields. |
| `test_chunk_state.py` *(extended)* | Offset ranges, `get_tokens_processed` excluding the offset, an offset plus a sub-chunk suffix staying inactive, `disable` clearing the offset. |
| `test_job.py` *(extended)* | `past_seen_tokens` with a cached prefix during prefill and at the first decode step; `JobCache.next_write_point`, exercised through `job.caching`. |
| `test_job_tracker.py` *(extended)* | Reservations released on complete, cancel and stale expiry; the sweep expiring entries; a tracker built without a cache. |
| `test_model_manager.py` *(extended)* | Unloading layers or an end model drops that process's entries; unloading without the hook still works. |
| `test_oai_responses.py` *(extended)* | `input_tokens_details` / `prompt_tokens_details` shape on both endpoints, streaming and not. |
| `test_config.py` *(extended)* | Defaults when absent from an existing file, round-trip through save/load, `0` surviving the round trip, `to_string`. |
| `test_jobs_server_page.py` *(new)* | Navigation across all seven rows and both wraps, digit editing that persists, `0` accepted, enter routing to the keys page and the start row, the rendered rows, tips, and the stats line. |

### Exit criteria

- ✅ 468 unit tests pass (348 before), no skips added. The only existing tests
  changed were the ones being extended, plus a `cache_options=None` parameter on the
  `complete_cb` fakes in `test_oai_responses.py` / `test_oai_cancel.py` and
  `process_id` / `prompt_cache` on the fakes in `tests/…/unit/util.py`. The `JobCache`
  extraction later moved those assertions onto `job.caching` and gave
  `test_oai_responses.py`'s `DummyJob` a `JobCache` instead of a bare `cached_tokens`.
- ✅ `ruff check src tests` reports the same 60 pre-existing findings, none in the
  new files.
- ⚠️ The integration runs were **not** executed: they need model weights.
  `job_processor/test_prompt_cache_path.py::TwoRequestTests` stands in for the
  single-node case (the second of two identical requests adopts 512 tokens and
  embeds strictly fewer chunks) and for the disabled case (`max_cache_time = 0`
  leaves the embed count, the entry count and the reservation total exactly as an
  uncached run). Before Phase 2, still run against real weights:
  - two requests sharing a ≥ 256-token prefix on a single-node pipe — the second
    reports `cached_tokens ≥ 256`, returns the same text at `temperature = 0`, and
    has a lower time-to-first-token;
  - the same with a Gemma 3 or Qwen3.5 end model, since snapshot-not-crop is only
    exercised by sliding-window and linear-attention stacks;
  - the two-node case, which must still report `cached_tokens == 0`.

---

## 4. Phase 2 — distributed reuse ✅

**Goal.** Remove the local-pipe gate by giving layer nodes the tags and the
back-channel from §5. This is the phase with the most integration risk; every step
below is written so it can be tested with the fakes in `tests/language_pipes/unit/util.py`
before touching a real pipe.

### Steps

**2.1 — Wire fields.** (S)

- `NetworkJob` appends, after `pass_idx` (Phase 0 fixed its slot): `attempt`,
  `cache_use_id`, `cache_use_tokens`, `cache_write_id`, `cache_write_tokens`,
  `cache_reserve_tokens`. Defaults `0` / `b''`.
- `attempt` is new here. It belongs with the rest of the pass bookkeeping, so put it
  on `PassSequence` (`jobs/pass_sequence.py`) rather than on `Job` — and the five cache
  tags belong on `JobCache` (`jobs/job_cache.py`) for the same reason: `attempt: int = 0`,
  bumped by the origin on a `MISS` rebuild (2.5), plus a `reset()` that clears `idx`,
  `last_idx`, `key`, `outputs` and `retries` in one place. Keep the Phase 0 split —
  `PassSequence` decides, `Job` applies.
- In `Job.receive_network_job`, ahead of the Phase 0 `PassSequence.accept` call:
  `network_job.attempt > passes.attempt` → replace `self.cache` with a fresh
  `DynamicCache(config)`, `passes.reset()` to the new attempt, then continue;
  `network_job.attempt < passes.attempt` → return `False`. This needs the `config`
  kept on the `Job` (passed to `__init__` today but not stored). Tests in `test_job.py`:
  higher attempt rebuilds and resets the pass bookkeeping, lower is dropped, equal is
  unchanged.
- `Job.to_network_job()` emits them from `job.caching` and `job.passes`; the origin
  clears `cache_write_id` after the pass that carried it. `cache_use_id` /
  `cache_reserve_tokens` are sent only on the job's first packet
  (`compute_step == TOKENIZE → EMBED` transition, i.e. from `JobFactory.start_job`'s
  dispatch and the first `_state_send`); a `JobCache.first_packet_sent` flag is enough.
- `Job.receive_network_job` copies `cache_write_id` / `cache_write_tokens` into
  `job.caching.pending_write_id` / `pending_write_tokens` — a `JobCache.tag_from_wire`
  next to the origin's own `tag()` — so the Phase 1 store hook in
  `_state_process_layers` works unchanged on a layer node.
- Tests: `test_network_job.py` — round-trip with every tag; old-writer payload parses
  with empty tags. `test_job.py` — receive copies the write tag; a packet without it
  clears the pending tag.

**2.2 — `CacheStatus` packet and dispatch.** (S)

- `src/language_pipes/jobs/cache_packets.py`: `CacheStatus(job_id, pipe_id, attempt,
  reason)` with `reason` an `IntEnum {MISS, NO_STORE, ABORT}`, `to_bytes` /
  `from_bytes` mirroring `jobs/job_cancel.py`.
- `CACHE_PROTOCOL = 3` in `job_receiver.py` next to `CANCEL_PROTOCOL`;
  `ContentProvider._receive_data` adds the branch;
  `JobReceiver._send_cache_status(node_id, status)` mirrors `_send_cancel`
  (including the local-loopback case) and `JobReceiver.receive_cache_status(node_id,
  data)` parses and routes by reason.
- Tests: round-trip; `test_content_provider_routing.py` — protocol 3 reaches the
  receiver; malformed bytes are ignored.

**2.3 — Layer-node read path and admission.** (M)

- `JobTracker.add_job` grows a result: it needs to say *hit*, *miss*, or *no
  budget*, and the receiver — the only thing that can send — acts on it. Recommended
  shape: `add_job` returns `Tuple[Optional[Job], CacheOutcome]`, or a small
  dataclass, rather than raising.
  - If `network_job.cache_use_id != b''`: `lookup(cache_use_id,
    network_job.origin_node_id, model_id, process_id, start_layer, end_layer)`. The
    layer range and `process_id` come from the local `LlmModel` for this pipe
    (`pipe.get_layer(network_job.current_layer, need_physical=True)`), so the
    receiver passes them in — or, better, let `CachePolicy.identity(job)` return the
    layer-node shape when this node is not the origin, so the tracker and the
    processor's write path stay on one definition. Hit → `adopt` into the new job's `cache`, and
    `job.caching.adopt(blocks)`. Miss → do **not** add the job; return `MISS`.
  - If `cache_reserve_tokens > 0`: `reserve(job_id, cache_reserve_tokens)`; on
    refusal return `NO_STORE` alongside the job (the job still runs).
- `_process_network_job` (split out of `_job_runner_loop` in Phase 0): on `MISS`, send `CacheStatus(MISS, attempt)` to
  `network_job.origin_node_id` and return without processing. On `NO_STORE`,
  send it and proceed.
- Tests: `test_job_tracker.py AddJobTests` — hit adopts (job cache is the adopted
  container), miss returns no job, origin mismatch is a miss, reservation refusal
  reports `NO_STORE` but still adds the job.

**2.4 — Layer-node write path.** (S)

- Nothing new in the processor: the Phase 1 hook in `_state_process_layers` fires on
  `job.caching.pending_write_id`, which 2.1 populates from the packet. Confirm the snapshot
  validation runs on layer nodes with a test in `test_state_layers.py` that builds a
  job through `receive_network_job` rather than `make_job`.
- `store` on a layer node needs `origin_node_id`, `model_id`, `process_id`, layer
  range — all available from the job and the local `LlmModel`.

**2.5 — Origin handling of `MISS`, `NO_STORE`, `ABORT`.** (M)

- `MISS` (attempt matches the job's current attempt):
  1. `job.passes.reset()` with the bumped attempt (which clears `idx`, `last_idx`
     and `outputs` together); `job.cache = DynamicCache(config)`;
     `job.caching.forget()` (which drops the ids, the write points and any pending
     tag) and `job.caching.prefix_len = 0`; `caching.options.enabled = False` for the
     rest of this job.
  2. Broadcast `CacheStatus(ABORT, old_attempt)` to every distinct `node_id` in the
     pipe's segments other than the origin.
  3. Re-run from the tokenize branch: `prompt_tokens` and `input_ids` are still
     valid; reset `chunking` to offset 0 and set `compute_step = EMBED`, then queue
     the job back through the runner. This is a *rebuild*, distinct from the Phase 0
     bounce *replay*: the replay resends a saved pass against unchanged caches, the
     rebuild discards every cache and starts the prefill over. Keep them as two
     functions with those names so the difference stays visible.
- `MISS` for an attempt lower than the job's current one: ignore (log debug).
- `NO_STORE`: clear the job's remaining write points; the next `to_network_job` sends
  no `cache_write_id`. Entries already stored by other nodes for earlier boundaries
  of this job stay — they are complete across the pipe because the refusing node
  either stored them too (it refused later) or never reserved (it refused at
  `add_job`, before any write point). Only the latter can happen, since reservation
  is decided once at `add_job`; state that in a comment.
- `ABORT` on a layer node: if `attempt` matches the local job's `passes.attempt` →
  `remove_job` + `_drop_queued` + `release`; if lower → ignore.
- Tests: `test_job_receiver.py` — the two orderings from §10 (late `ABORT` after the
  retry's packet is ignored; `MISS` naming a dead attempt does not abort the retry);
  `MISS` sends `ABORT` to every non-origin segment node exactly once. `job_processor/`
  — after a `MISS` restart, the second run has `caching.prefix_len == 0` and emits no
  `cache_use_id`.

**2.6 — Remove the local-pipe gate.** (S) Delete the `Pipe.is_local_to` check from
`CachePolicy.usable` (`jobs/cache_policy.py`; it moved off `_state_embed` at the end of
Phase 1 — keep the `Pipe` helper if the TUI uses it). Update the Phase 1 test that
asserted "no hit on a remote pipe" to assert a hit *with* tags emitted.

The steps above landed as written apart from the departures recorded below.

**2.7 — Docs.** (S) `documentation/architecture.md`: replace the Phase 1 "single-node
only" note with the §5 protocol summary (tags, `CacheStatus`, `attempt`); a row in the
failure-modes list for each of `MISS`, `NO_STORE`, `ABORT`. `documentation/oai.md`:
drop the single-node caveat. `configuration.md`: the note that each node applies its
own limits and a `0` on any node disables reuse across pipes through it.

### Decisions that differ from the plan as written

| Plan said | Shipped | Why |
|---|---|---|
| `add_job` returns `Tuple[Optional[Job], CacheOutcome]` or a small dataclass (§7 item 2). | The tuple, with `CacheOutcome` (`OK` / `MISS` / `NO_STORE`) in `jobs/cache_policy.py`. | Two call sites and one of them is a test. A dataclass would be a name for a pair that is never passed anywhere. |
| `add_job` does the lookup itself, taking layer range and `process_id` from the receiver. | `add_job` takes an optional `CachePolicy` and calls `policy.adopt_for_node(job, network_job)`. | The plan's own "or, better" option. The tracker gains no cache knowledge, and `identity()` stays the single definition of what a node's slice is - the alternative was the receiver deriving one shape and `store_tagged_pass` deriving another. |
| Layer-node identity is the segment hosting `network_job.current_layer`. | Every physical local segment of the pipe, sorted: process ids from all of them, `start_layer` the lowest, `end_layer` the highest. | `current_layer` has already moved on by the time the write path asks, so the two would disagree. The node's cache holds all of its segments anyway, and this shape is a pure function of node + pipe, so `add_job` and the store a pass later cannot diverge. |
| The origin's identity narrows to its own layers now that the gate is gone. | Unchanged from Phase 1: the whole pipe's process ids. | It costs nothing and buys free invalidation - if a segment moves to another node, the origin misses locally instead of adopting, tagging, and paying a `MISS` round trip to learn the same thing. |
| The Phase 1 store hook fires unchanged on layer nodes (2.4). | The hook moved. Phase 1 stored when the pass left `ComputeStep.LAYER`, which only ever happens on the node hosting the *last* layers; a middle node never stored. It now fires when this node has finished the last of *its* layers for the pass (`CachePolicy.pass_complete_here`), which covers the middle node, the origin whose own layers stop before the pipe's, and a node hosting two ranges of one pipe. | Without it a three-node pipe would store on one node out of three and miss forever after. |
| `store_tagged_pass` claims the tag (`take_pending`). | It reads it (`pending`); the tag is cleared once per pass by the next `tag()` or `tag_from_wire()`. | The tag has to survive the store: the packet carrying it goes on to the next node, which stores *its* slice under the same ID. Claiming it would mean only the first node on the pipe ever wrote. |
| The read tags ride the job's first *packet* (`first_packet_sent`). | They ride every packet of pass 1 (`passes.idx <= 1`). | A node that bounced the first packet has not created its job yet, so the resend is its only chance to be told what to adopt. With the flag it would have prefilled from token 0 while the rest of the pipe resumed from the prefix. |
| — | **The end-of-response entry was rewritten.** `CachePolicy.store_end_of_response` is gone; the boundaries the answer crosses are tagged as it crosses them, from `tag_write_point` during decode. | Two reasons, one of them a bug. (1) At completion no pass is left to carry a tag, so only the origin would have stored - the next request would adopt, every layer node would `MISS`, and multi-turn chat would rebuild every time. (2) The Phase 1 version stored a cache holding `len(input_ids) - 1` positions under an ID naming the block boundary *below* that, so unless the response happened to end exactly on a boundary the entry was longer than it claimed and a job adopting it resumed at the wrong offset. Tagging the pass whose end lands exactly on the boundary fixes both, and reuses the machinery that already existed. The cost is entries at each boundary the answer crosses rather than one at the end; the budget and the LRU bound it, and §6's per-block snapshots were already headed this way. |
| — | `receive_cache_status` checks which end of the job this node is. | `MISS` / `NO_STORE` are answers to a job this node dispatched and `ABORT` is an instruction about one it did not; without the check a stray packet would have a layer node rebuilding a job it does not own. |
| — | `Pipe.is_local_to` deleted rather than kept. | The plan said to keep it if the TUI used it. Nothing did. |
| — | `_state_embed` also asks to store, after the embed. | An origin that hosts the end model but none of the layers never enters `PROCESS_LAYERS`. It holds nothing to snapshot, but it is the only node that sees tokens, so without an entry of its own its next lookup finds nothing and no node on the pipe is ever told to adopt. On an origin that does host layers the call is a no-op - the pass has not reached the last of them yet. |

### Tests

| File | Covers |
|---|---|
| `test_network_job.py` *(extended)* | Every tag round-trips; a payload truncated to the pre-Phase-0 writer's length parses with `attempt == 0` and empty tags. |
| `test_job.py` *(extended)* | `JobAttemptTests`: a higher attempt drops the cache and the numbering, a lower one is dropped without canceling, an equal one behaves as before, the attempt goes on the wire, `rebuild()` returns the job to `TOKENIZE`. `JobCacheTagTests`: the write tag comes off the packet, an untagged packet clears the one before, the read tags are carried onward, they ride only the first pass, a replayed first pass still carries them, the origin ignores tags on its own returning packet. |
| `test_cache_protocol.py` *(new)* | `CacheStatus` round-trip; a layer node that cannot adopt reports `MISS` and computes nothing; a layer node with no budget reports `NO_STORE` and runs anyway; the origin's rebuild - new attempt, no prefix, no tags on the retry, prefill from token 0, `ABORT` to every other segment node once, reservation released; a `MISS` naming a dead attempt changes nothing; `NO_STORE` stops the writes without restarting; an `ABORT` that lost the race against the retry is ignored; queued packets for an aborted job are dropped; the origin/layer-node guards; unknown job and unparseable payload. |
| `test_job_tracker.py` *(extended)* | `LayerNodeReadPathTests`: a hit adopts into the new job, a miss adds no job, another origin's entry is a miss, a length that disagrees with the origin is a miss, a node hosting none of the pipe is a miss, a refused reservation reports `NO_STORE` and still runs, an untagged packet touches the store not at all. |
| `test_content_provider_routing.py` *(extended)* | Protocol 3 reaches `receive_cache_status` and nothing else. |
| `job_processor/test_state_layers.py` *(extended)* | `LayerNodeWriteTests`, built through `receive_network_job`: the slice is stored under the ID the origin named, the tag is forwarded on, a drifted pass is not stored, a node with a second range of the same pipe waits for its last visit. |
| `job_processor/test_distributed_cache.py` *(new)* | Two nodes, two requests, through the real wire format and the real `add_job`: both nodes store the prompt boundary under the same ID; the second request adopts on both and embeds only the tail; both caches end the same length; the read tags ride one pass; a relay whose entries are gone reports `MISS`; `max_cache_time = 0` on the relay alone leaves the request correct and uncached. `GatewayOriginTests` covers the origin that hosts no layers at all. |
| `job_processor/test_prompt_cache_path.py` *(extended)* | The Phase 1 "no reuse on a remote pipe" test now asserts the tags are emitted instead; a miss still asks the other nodes to reserve; the store no longer consumes the tag; `DecodeWritePointTests` replaces `EndOfResponseTests` - the pass that lands on a boundary is tagged and the one that lands short is not, each boundary is planned once, an uncached job and a `NO_STORE`d job are never tagged, and the stored entry is exactly as long as it claims. |

### Exit criteria

- ✅ 529 unit tests pass (473 before), no skips added. The only existing tests
  changed were the ones being extended, plus the `add_job` tuple in
  `test_job_tracker.py`.
- ✅ `ruff check src tests` reports the same 60 pre-existing findings, none in
  the new files.
- ⚠️ The integration and manual runs below were **not** executed: they need model
  weights. `job_processor/test_distributed_cache.py` stands in for the two-node
  case, including the forced miss, and asserts the invariant the design rests on
  (both nodes end every request holding the same number of positions). Still to
  run against real weights before Phase 3 leans on any of it:
  - two-node and three-node cases (`test_double_node`, `test_triple_node` in
    `integration/oai.py`) extended with the prefix-sharing pair; the second
    request reports `cached_tokens > 0` and matches the uncached output at
    `temperature = 0`;
  - the two-node hand-driven restart check that Phase 0 deferred, which Phase 2
    now leans on for drift detection;
  - forced-miss run: `max_cache_time = 0` on one layer node only; the second
    request succeeds with `cached_tokens == 0` and the origin log shows exactly
    one `MISS` → rebuild;
  - forced-`NO_STORE` run: `max_cache_tokens` on one layer node below the job's
    estimate; the request succeeds, no node stores, no later request through that
    pipe rebuilds;
  - a multi-turn run, to confirm the decode-boundary entries make the second turn
    hit across nodes;
  - mixed-version run (manual, once): one node on `main` before this branch.
    Requests succeed uncached.

---

## 5. Phase 3 — full OpenAI surface ✅

**Goal.** §2.1 in full, §4.5, explicit-mode write points from §4.3,
`cache_write_tokens`, and chat-completion `stream_options.include_usage`. All origin
side.

### Steps

**3.1 — Parameter parsing and validation.** (S)

- `oai_cache.parse_cache_options` reads `prompt_cache_options.mode` / `.ttl`,
  `prompt_cache_retention` (alias, `"in_memory"` → default TTL, `"24h"` → 86400), and
  `prompt_cache_breakpoint` on `input_text` content blocks (message index recorded).
- Validation errors raise a `ValueError` subclass that `oai_responses_create` turns
  into a `400` via `_send_code`: unknown `mode`, unknown `ttl`, unknown retention, a
  breakpoint on top-level `instructions`, more than 4 breakpoints (extras ignored, not
  an error — match §2.1).
- Effective TTL = `min(requested, max_cache_time)`, resolved when the entry is stored
  (the node's limit is read then).
- Tests: `test_oai_responses.py` — each 400; alias mapping; chat endpoint ignores
  `prompt_cache_options` silently (§2.2).

**3.2 — Breakpoint → token offset.** (M)

- `EndModel.prefix_token_counts(messages, indices) -> List[int]`: for each marked
  message index `k`, render `messages[0:k+1]` with `add_generation_prompt=False`,
  tokenize, and return the length. Lives next to `EndModel.tokenize`
  (`end_model.py:77`) because that is where the tokenizer and chat template are.
- In `_state_embed`'s tokenize branch: for each candidate `L_k`, accept only if
  `input_ids[:L_k] == prefix_tokens` (the "not a real prefix" check); round down to a
  block boundary; drop `< MIN_CACHE_TOKENS`; keep at most 4; sort ascending. These
  become `caching.write_points` in explicit mode, replacing the implicit ones
  (`JobCache.plan_prompt_write` is the implicit-mode counterpart). If none
  survive in explicit mode, log a warning and disable caching for the job.
- Tests: with a `FakeEndModel` that returns scripted prefix lengths — accepted,
  rejected (not a real prefix), rounded, capped at 4, below minimum dropped; explicit
  mode with zero survivors disables the job's caching.

**3.3 — `cache_write_tokens`.** (S)

- `JobCache.write_tokens` accumulates the token count of every entry the *origin*
  stored for this job (layer-node stores are not observable and are assumed to
  match). Reported in `input_tokens_details.cache_write_tokens` for Responses only
  (OpenAI does not expose it on chat completions).
- Tests: shape in streaming and non-streaming; value equals the write point tagged
  in a single-boundary run.

**3.4 — Chat `stream_options.include_usage`.** (S)

- `ChatCompletionRequest.from_dict` reads `stream_options.include_usage`; the chat
  streaming path in `oai_chat_complete` emits one final chunk with `choices: []` and
  a `usage` block, then `[DONE]`, only when requested.
- Tests: final chunk present iff requested; usage matches the non-streaming values.

**3.5 — Docs.** (S) `documentation/oai.md` "Prompt Caching": the full parameter
table from §2.1, an explicit-mode example with a breakpoint, `cache_write_tokens`,
and `stream_options.include_usage`. Remove the "arrives in a later release" note from
Phase 1.

The steps above landed as written apart from the departures recorded below.

### Decisions that differ from the plan as written

| Plan said | Shipped | Why |
|---|---|---|
| `EndModel.prefix_token_counts(messages, indices) -> List[int]`, and `_state_embed` checks `input_ids[:L_k] == prefix_tokens`. | `EndModel.prefix_tokens(messages, indices) -> List[List[int]]` - the renderings themselves - and `JobCache.plan_explicit_writes(prefixes, input_ids)` does the check, the rounding, the minimum and the ordering. | The caller cannot check a prefix it was given only the length of, so the two halves had to meet somewhere. Putting the rule on `JobCache` keeps it with the other decisions that read only the job's own state, and it makes the "not a real prefix" case testable against a scripted fake instead of a real tokenizer. |
| The mapping happens "in `_state_embed`'s tokenize branch". | `CachePolicy.plan` calls `_plan_explicit` before admission. | Phase 1 moved the whole read/write path off the FSM; `_state_embed` no longer knows what a write point is. Running before `reserve` also means a job with nothing to write never holds a reservation for it. |
| `parse_cache_options(data, authenticated)` grows the new parameters. | `parse_cache_options(data, authenticated, responses=False)`. | §2.2 says chat completions take only `prompt_cache_key`. Without the flag the chat endpoint would either 400 on a parameter it does not document or need its own parser. |
| `JobCache.write_tokens` accumulates the token count of every entry the origin stored. | It accumulates what each entry *adds*: `tokens - max(previous write, prefix_len)`. | The raw counts double-count. §2.4's own example is a 4096-token prompt with `cached_tokens: 3968` and `cache_write_tokens: 128`, and with the decode-boundary entries Phase 2 added, the raw sum for one request would exceed its own total tokens several times over. The two readings coincide on the single-boundary run §3.3 asks the test to assert. |
| — | `ttl` does not ride the wire. | Phase 3 is origin-side, and the tags are fixed by Phase 2. A layer node therefore keeps its slice for its own `max_cache_time`. A *shorter* requested TTL then expires the origin's entry first, so the pipe misses locally and never adopts a prefix a node is missing - the conservative direction, at the cost of some memory held on layer nodes until it idles out. Documented in `oai.md`. |
| — | `"in-memory"` is accepted alongside `"in_memory"`. | Both spellings are in circulation; a hyphen is not an unknown value in the sense §2.1 means to reject. |
| — | `_response_input_to_messages` returns `(messages, sources)`. | A breakpoint marks an `input` *item*, the write path needs a *message* index, and `instructions` plus the tool schemas insert messages ahead of both. The item-to-message map is the only honest way to line them up, and it also covers items that produce no message at all. |
| — | A breakpoint is honored on a tool result's `output` blocks, not only on a message's `content`. | Those are the two places a Responses item keeps content blocks, and a long tool result is exactly the stable block a client would mark. |
| — | The parsing surface got its own test file (`test_prompt_cache_options.py`) rather than growing `test_oai_responses.py`. | It is one subject - what the server accepts, rejects and ignores - and `test_oai_responses.py` is already 900 lines about a different one. |

### Tests

| File | Covers |
|---|---|
| `test_prompt_cache_options.py` *(new)* | `mode`, `ttl` and `prompt_cache_retention` parsing, the alias mapping and which spelling wins; every 400 (unknown mode, unknown ttl, unknown retention, a non-object `prompt_cache_options`, a non-object or unknown-mode breakpoint, a breakpoint on `instructions`); breakpoints capped at four; item-to-message index mapping through `instructions`, tool schemas, a tool result and an item that produced no message; the chat endpoint ignoring all of it; and the four 400s again over real HTTP, with a valid explicit request still returning 200. |
| `job_processor/test_prompt_cache_path.py` *(extended)* | `ExplicitModeTests`: a breakpoint becomes a write point rounded down to a block; the implicit end-of-prompt boundary is not added; several breakpoints keep their order; a rendering the prompt does not start with is dropped; one below the minimum is dropped; no survivor runs the job uncached with no reservation; a boundary the adopted prefix covers is dropped; the answer crossing a boundary is not tagged. `CacheWriteTokensTests`: a single boundary counts the whole entry, an adopted prefix is not counted, a later boundary counts only what it adds, a skipped store counts nothing, and a layer node counts nothing. |
| `test_oai_responses.py` *(extended)* | `cache_write_tokens` in `input_tokens_details`, streaming and not. `ChatStreamUsageTests`: no usage unless asked for, one usage chunk with empty `choices` immediately before `[DONE]`, the numbers matching `chat_usage`, `include_usage: false` respected, and a malformed `stream_options` ignored. |

### Exit criteria

- ✅ 575 unit tests pass (529 before), no skips added. The only existing tests
  changed were the three `input_tokens_details` assertions that now carry
  `cache_write_tokens`, plus `prefix_tokens` on `FakeEndModel`.
- ✅ `ruff check src tests` reports the same 60 pre-existing findings, none in
  the new files.
- ⚠️ The integration run was **not** executed: it needs model weights.
  `ExplicitModeTests` stands in for it at the FSM level. Still to run against
  real weights: an explicit-mode request with one breakpoint on a long system
  message, followed by a request with a different user turn - the second hits at
  the breakpoint boundary and not beyond, and `cache_write_tokens` on the first
  equals that boundary.

---

## 6. Phase 4 — optimizations (each independent, each optional)

| Item | Where | What it needs | Note |
|---|---|---|---|
| Adopt-by-move under budget pressure (§4.7) | `PromptCache.reserve` / `adopt` | An entry refcount (jobs currently sharing it); when 0 and admission would refuse, hand the tensors over and delete the entry, and drop `caching.prefix_len` from the estimate. | Do first: it is the cheapest win on small nodes. |
| CPU demotion of cold entries (§4.2) | `PromptCache.sweep` | After one TTL period unused, `.to("cpu")` each tensor; `adopt` moves back. `size_gb` stays; a `device` field is added to `CacheEntry`. | Only worth it when VRAM, not tokens, is the binding limit. |
| Per-block snapshots (§11) | `_state_embed` write-point planning | Tag every block boundary inside the prompt, not just the last; entries share tensors so the cost is bookkeeping, not memory. | Makes a request that diverges mid-prompt still hit on the shared head. |
| Derived default for `max_cache_tokens` (§3.1, §12.5) | `config.py`, `model_provider` | `bytes/token` from the hosted model's config and layer count, budget = fraction of the node's configured memory. | Keep the explicit field; derive only when it is absent from the file. |

---

## 7. Cross-cutting decisions to settle before Phase 1 starts

All three are settled: **default on** (300 / 16384) and an **instance-level
secret** with Phase 1, and the `add_job` return shape with Phase 2.

1. **Default on or off in Phase 1.** ✅ *Shipped default-on.* `cache_plan.md` sets `max_cache_time = 300` and
   `max_cache_tokens = 16384` as defaults, which turns the feature on for every
   upgraded node. With the §2.3 anon rule, a default-on node only caches for callers
   who opt in with `prompt_cache_key` or who authenticate, which is a reasonable
   default. If the reviewer would rather soak Phase 2 first, ship Phase 1 with
   `DEFAULT_MAX_CACHE_TIME = 0` and flip it in the Phase 2 release; nothing else
   changes.
2. **`add_job` return shape in 2.3.** ✅ *Shipped the tuple*, `(Optional[Job],
   CacheOutcome)`. Two call sites, one of them a test; a dataclass would be a
   name for a pair that is never passed anywhere.
3. **Where the per-process secret lives.** ✅ *Shipped instance-level.* Module-level in `prompt_cache.py`
   (simplest; one per process) or on the `PromptCache` instance (one per
   `set_router`, so restarting the network in the TUI without restarting the process
   also rotates it). Instance-level is slightly more conservative and costs nothing;
   recommended.

---

## 8. Verification checklist per phase

Run before opening the PR for each phase:

```
pytest                                   # unit suite, what CI runs
python -m tests.language_pipes.unit      # same suite, unittest runner
ruff check src tests                     # dev extra in pyproject.toml
```

Then the phase's integration run from `tests/language_pipes/integration/oai.py`
(needs `Qwen/Qwen3-0.6B` downloaded; see CONTRIBUTING.md), and the manual runs
named in that phase's exit criteria. Record the observed `cached_tokens` and
time-to-first-token numbers in the PR description so the Phase 2 and Phase 3 PRs have
a baseline to compare against.
