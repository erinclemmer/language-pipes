# Prompt Caching Design

Status: proposal / implementation plan
Scope: KV prefix caching for the OpenAI-compatible server, with the request/response
surface matching OpenAI's Responses API (`POST /v1/responses`).

---

## 1. What we are building

"Prompt caching" here means the same thing it means at OpenAI: when a request's
rendered prompt **starts with a prefix that a previous request already processed**,
skip the prefill for that prefix and reuse the key/value state that was computed
for it. Reused tokens are reported back to the client as `cached_tokens`.

The single rule everything follows from is unchanged from OpenAI's: a hit
requires an **exact match of the whole token prefix up to a breakpoint**. One
different token anywhere before the breakpoint means everything after it is a miss.

What makes this project different from a single-box server is that **the KV
cache is not in one place**. Today (`documentation/architecture.md`, "KV cache
handling across nodes") every node holds `Job.cache` for only the layers it
hosts, and that cache is created in `Job.__init__` (`src/language_pipes/jobs/job.py:111`)
and thrown away when the job leaves `JobTracker.jobs_pending`. So a cached prefix
is a **distributed object**: for a prefix of N tokens to be reusable, *every*
node on the pipe must still hold its own slice of KV for exactly those N tokens.
The design below is mostly about keeping those slices in agreement.

Non-goals for this plan: `store` / `previous_response_id` server-side conversation
state, disk-backed caches, cache sharing between API keys, batching.

---

## 2. API surface (OpenAI-compatible)

### 2.1 Request parameters — `/v1/responses`

Parsed in `ResponsesRequest.from_dict` (`src/language_pipes/util/oai.py:207`).

| Parameter | Type | Behavior |
|---|---|---|
| `prompt_cache_key` | string | Scopes the cache. Requests with the same key + same prefix share entries. Optional; defaults to `""`. |
| `prompt_cache_options.mode` | `"implicit"` \| `"explicit"` | `implicit` (default): a write point is placed at the end of the prompt and at the end of the response. `explicit`: only client-marked breakpoints are written. |
| `prompt_cache_options.ttl` | `"30m"` | Requested lifetime. Clamped to the node's `max_cache_time` (§3). |
| `prompt_cache_retention` | `"in_memory"` \| `"24h"` | Older-model spelling; accepted as an alias and mapped to seconds, then clamped the same way. |
| `prompt_cache_breakpoint` | object on an `input_text` content block | `{"mode": "explicit"}` marks the end of a stable block as a write point. Max 4 per request; extras ignored. |

Validation: unknown values (e.g. `mode: "auto"`, `ttl: "5h"`) return `400` with the
usual `_send_code` error body, matching OpenAI's behavior of rejecting bad enum
values rather than silently ignoring them. A breakpoint attached to top-level
`instructions` is rejected with a 400 that says to move the text into an `input`
developer message — same as OpenAI.

### 2.2 Request parameters — `/v1/chat/completions`

Only `prompt_cache_key` is accepted (that is all OpenAI exposes there). Caching
otherwise behaves as `mode: "implicit"`.

### 2.3 Response fields

Responses (`_response_json`, `src/language_pipes/util/oai.py:297`):

```json
"usage": {
  "input_tokens": 4096,
  "input_tokens_details": { "cached_tokens": 3968, "cache_write_tokens": 128 },
  "output_tokens": 210,
  "total_tokens": 4306
}
```

Chat completions (`src/language_pipes/util/oai.py:409`):

```json
"usage": {
  "prompt_tokens": 4096,
  "prompt_tokens_details": { "cached_tokens": 3968 },
  "completion_tokens": 210,
  "total_tokens": 4306
}
```

`cached_tokens` is always a multiple of the block size (§4.1), which matches
OpenAI's documented rounding on pre-5.6 models, so clients that assume the
rounding are not surprised. Streaming reports the same numbers: the Responses
stream already carries `usage` in the `response.completed` event; chat-completion
streams gain a final chunk with `usage` when the client sends
`stream_options.include_usage` (added at the same time, since it is the only
OpenAI-sanctioned way to get usage out of a chat stream).

There is deliberately **no** endpoint to inspect or clear the cache, matching
OpenAI. Operators see totals in the TUI instead (§10).

---

## 3. Configuration

One new field on the jobs server, as requested.

`src/language_pipes/config.py` (alongside `max_node_jobs` / `max_api_jobs`):

```toml
# Maximum seconds a cached prompt prefix is kept after its last use. 0 disables
# prompt caching on this node.
max_cache_time = 300
```

- `DEFAULT_MAX_CACHE_TIME = 300` (5 minutes), stored in seconds as an int.
- `0` disables caching completely on that node: no reads, no writes, no memory held.
- Read/written through `JobProvider.get_max_cache_time` / `set_max_cache_time`
  (`src/language_pipes/content_provider/job_provider.py`), like the other two limits.
- Effective TTL for a request = `min(requested_ttl_or_default, max_cache_time)`.
  A client asking for `"24h"` on a node configured with 300s gets 300s; the
  response is not an error, since OpenAI treats retention as a request, not a contract.
- The lifetime is measured **from last use**, not from creation — reusing a prefix
  refreshes it, matching OpenAI ("a busy prefix stays warm, an idle one expires").
- Every node applies its own `max_cache_time` to its own slice. Nodes are not
  required to agree; disagreement just makes hits rarer (§7), never wrong.

TUI: a fourth editable row on the Jobs / Server page
(`src/language_pipes/tui/components/jobs_server/top_state.py`), inserted after
"Max API Jobs" — `focus_idx` 3 becomes "Max Cache Time", and the api-keys /
start-server rows shift to 4 and 5. Needs a matching `TIPS["jobs_server"]["max_cache_time"]`
entry in `src/language_pipes/tui/frame/tips.py`:

> Max Cache Time: How long (in seconds) a processed prompt prefix is kept in
> memory so a follow-up request that starts with the same text can skip
> re-processing it. Set to 0 to disable prompt caching.

Docs to update: `documentation/configuration.md` (new `max_cache_time` section
under "API Server"), `documentation/oai.md` (a "Prompt Caching" section covering
§2), `documentation/architecture.md` (the KV-cache section currently says caches
die with the job).

---

## 4. Core mechanism

### 4.1 Block-chained prefix IDs

Prefix identity is a hash chain over fixed-size token blocks, so that two
requests that share the first K blocks produce the same first K chain values.

```python
BLOCK_SIZE = 128          # tokens; must be a multiple of CHUNK_SIZE (32)
MIN_CACHE_TOKENS = 256    # 2 blocks; shorter prefixes are never cached

scope = sha256(b"lp-prompt-cache-v1" + origin_node_id + api_key + prompt_cache_key)
h[0] = scope
h[i] = sha256(h[i-1] + tokens[(i-1)*BLOCK_SIZE : i*BLOCK_SIZE] as int32 LE)
```

`h[i]` is the 32-byte ID of the prefix that is exactly `i * BLOCK_SIZE` tokens long.

Properties worth calling out:

- **The chain is computed only on the origin**, because only the origin ever sees
  tokens (`EndModel.tokenize`). Layer nodes receive opaque 32-byte IDs and never
  learn anything about the prompt from them.
- The scope includes `origin_node_id` and the API key, so **entries are never
  shared across users or across origin machines**. This is a privacy decision,
  not a performance one: a cross-tenant hit is an oracle that tells one user
  another user sent a particular prefix. It also means a shared layer node cannot
  correlate two different users' prompts (§8).
- `prompt_cache_key` further partitions within one API key, which is what clients
  use to keep unrelated workloads from evicting each other.
- Block size 128 gives `cached_tokens` the same 128-granularity OpenAI reports,
  and divides evenly into `CHUNK_SIZE = 32` so cache boundaries always fall on
  prefill chunk boundaries.

### 4.2 Node-local store

New file `src/language_pipes/jobs/prompt_cache.py`:

```python
@dataclass
class CacheEntry:
    cache_id: bytes          # h[i]
    model_id: str
    process_id: str          # LlmModel/EndModel process id: invalidates on reload
    start_layer: int
    end_layer: int
    token_count: int         # i * BLOCK_SIZE
    cache: DynamicCache      # this node's slice only
    size_gb: float
    created: float
    last_used: float
    expires_at: float        # last_used + min(requested_ttl, max_cache_time)

class PromptCache:
    def lookup(self, cache_id, model_id, process_id, start_layer, end_layer) -> Optional[CacheEntry]
    def adopt(self, entry) -> DynamicCache       # deep copy for the job to mutate
    def store(self, cache_id, job, segment, token_count, ttl) -> None
    def sweep(self) -> None                      # TTL + LRU eviction
```

One `PromptCache` per node, owned by `ContentProvider` next to `JobTracker`, so
both the origin path (end model slice) and the layer path see the same store.

- **Adoption copies.** Two concurrent jobs can hit the same entry, and a job
  mutates its cache as it decodes, so `adopt` clones the key/value tensors rather
  than handing out the shared one. A clone of a few thousand tokens of KV is
  milliseconds; it also makes rollback on a miss free (just drop the copy).
- **Entries stay on the compute device.** Moving to CPU would save VRAM but give
  back much of the latency win on adopt. If VRAM pressure turns out to dominate,
  an entry can be demoted to CPU after one TTL period without changing anything else.
- **Eviction** is TTL-first, then LRU against a byte cap. Start with module
  constants `PROMPT_CACHE_MAX_GB = 4.0` and `PROMPT_CACHE_MAX_ENTRIES = 64`,
  measured with the same tensor walk as `Job.get_job_ram` (`src/language_pipes/jobs/job.py`).
  The sweeper runs on the existing 10s `JobTracker.check_stale_jobs` cadence
  rather than adding a thread, and calls the same `gc.collect()` /
  `torch.cuda.empty_cache()` / `malloc_trim` sequence that job cleanup already uses.
  If the in-flight node-memory work lands a configured memory ceiling, the cache
  cap should be derived from it instead of a constant.

### 4.3 Write path — snapshot points

A node can snapshot its slice at a boundary only if its cache covers exactly that
many tokens at that moment. That happens naturally right after the pass whose
last token is the boundary. So writes are **piggybacked on the job packet** and
need no extra round trip:

1. The origin knows, before it embeds a chunk, whether that chunk ends on a write
   point (§4.5). If it does, it tags the outgoing `NetworkJob` with
   `cache_write_id = h[i]` and `cache_write_tokens = i * BLOCK_SIZE`.
2. Every node that computes layers for that pass calls `PromptCache.store(...)`
   immediately after `process_job`, before forwarding.
3. The origin does the same for its own end-model layers in `_state_process_layers`
   (`src/language_pipes/jobs/job_processor.py:238`), then clears the tag so the
   next pass is untagged.

Because the tag rides the one packet that defines the pass, every node stores the
same boundary or none at all — no barrier, no consensus.

**Snapshot validation.** Before storing, a node checks that the pass it just ran
ends where the origin says it does: `job.data.cache_position[-1] + 1 == cache_write_tokens`.
On mismatch it skips the store (and logs at debug). This is cheap insurance against
a node whose cache silently drifted — notably the pre-existing hazard where
`JobReceiver.restart_token` re-runs a pass on nodes that already appended KV for
it, which appends duplicate entries. That hazard exists today; caching makes it
worth confirming and fixing separately, and the validation keeps a drifted node
from poisoning the shared prefix in the meantime.

Which boundaries get written:

| Mode | Write points |
|---|---|
| `implicit` (default) | Largest block boundary ≤ prompt length, and largest block boundary ≤ total length at completion (prompt + generated). |
| `explicit` | Only client breakpoints, rounded down to block boundaries (max 4). Nothing is written for the tail. |

The end-of-response write is what makes multi-turn chat hit: the next request's
prompt is the previous prompt + the generated answer + the new user turn, so the
stored entry is a strict prefix of it.

### 4.4 Read path — optimistic adopt, miss restarts

1. After `EndModel.tokenize` in `_state_embed` (`src/language_pipes/jobs/job_processor.py:208`),
   the origin walks its chain from the longest candidate down and asks its own
   `PromptCache` for a hit. The reusable length is capped at the largest block
   boundary ≤ `prompt_tokens - 1` — at least one token must remain to embed.
2. On a local hit the origin adopts the copy into `job.cache`, sets
   `job.cached_prefix_len`, and tags the first outgoing `NetworkJob` with
   `cache_use_id` / `cache_use_tokens`.
3. Each layer node handles the tag in `JobTracker.add_job`
   (`src/language_pipes/jobs/job_tracker.py:127`), which is where its `Job` (and
   its `DynamicCache`) is created today. Hit → the adopted copy becomes `job.cache`.
   Miss → the node does **not** process the packet; it replies `CACHE_MISS` to the origin.
4. On `CACHE_MISS` the origin broadcasts `CACHE_ABORT` to every segment node
   (each drops the job via `remove_job` + `_drop_queued`), then re-runs the same
   `job_id` from `ComputeStep.TOKENIZE` with reuse disabled for the rest of that job.
   Wasted work is bounded by one 32-token chunk through part of the pipe.

Optimistic rather than a prepare/ACK barrier because the common case is a hit —
all nodes wrote the entry on the same pass and expire it on similar timers — and a
barrier would put a round trip in front of *every* request to save a chunk of
compute on the rare one. The correctness of the miss path does not depend on
timing: a node that cannot adopt refuses to compute, so a lost or slow packet
degrades to a stalled job that the existing 60s expiry reaps, never to wrong output.

A later optimization (§12, phase 4) is to have each node publish a digest of the
IDs it holds into the DSN shared state, so the origin can skip an attempt that is
going to miss. IDs are unguessable without the API key, so publishing them leaks
nothing beyond cardinality.

### 4.5 Breakpoints → token offsets

A client marks a content block, but we need a token offset. At tokenize time:

1. Render and tokenize the full prompt as today (`apply_chat_template(...,
   add_generation_prompt=True)`).
2. For each marked block, render `messages[0:k+1]` with `add_generation_prompt=False`
   and tokenize → candidate length `L_k`.
3. Accept `L_k` only if the full token list actually starts with those `L_k`
   tokens. Chat templates are append-only in practice, but a template that
   rewrites earlier turns would silently produce a wrong offset, and this check
   catches it.
4. Round each accepted offset down to a block boundary; drop offsets below
   `MIN_CACHE_TOKENS`; keep at most 4.

If every breakpoint is rejected in `explicit` mode, the request simply does not
cache (matching OpenAI's "explicit mode with no breakpoints" behavior) and a
warning is logged.

### 4.6 Resuming prefill from a cached prefix

This is the part the existing code already almost supports.

`StaticAutoModel.compute_embedding` takes `past_seen_tokens` explicitly and
slices `input_ids[:, past_seen_tokens : past_seen_tokens + take]`, and
`PartialCacheMaskView` sizes the masks from that same number rather than from the
local cache. So resuming is a matter of making `past_seen_tokens` start at the
cached length:

- `Job.past_seen_tokens()` (`src/language_pipes/jobs/job.py:129`) becomes
  `self.cached_prefix_len + self.chunking.get_tokens_processed()` during prefill.
  The decode branch (`len(self.input_ids) - 1`) is unchanged.
- `ChunkState.init(prompt_length, start_offset=0)` gains the offset:
  `total_chunks` is computed from `prompt_length - start_offset`, and `get_range`
  returns `(start_offset + i*chunk_size, ...)`. `get_tokens_processed` keeps
  returning tokens covered *by this job's own chunks*, so the sum above is right.
- Layer nodes need no token bookkeeping at all. Positions, cache positions and
  masks all arrive in `JobData` from the origin; a node with a pre-populated cache
  and correct `cache_position` just works.

Edge cases:

- Cached length equals the prompt length (identical request replayed): capped to
  the largest boundary ≤ `prompt_tokens - 1`, so at least one token is embedded.
- Remaining suffix shorter than `CHUNK_SIZE`: `ChunkState` stays inactive and
  `compute_embedding` handles it in one pass — already the existing behavior.
- Sliding-window / hybrid stacks (Gemma 3, Qwen3.5 linear attention): entries are
  **snapshotted at a boundary, never cropped**, so a recurrent or windowed layer
  state is stored exactly as it stood at that token count. This is why the design
  does not use `DynamicCache.crop` to manufacture shorter prefixes from longer ones.
- Gemma 4 `shared_kv_states` ride in `JobData` and are recomputed per pass; they
  are not part of a stored entry.

---

## 5. Wire protocol changes

`NetworkJob` (`src/language_pipes/jobs/network_job.py`) gains four fields, appended
at the end of the serialization:

```
cache_use_id      bytes   (b'' when unused)
cache_use_tokens  int
cache_write_id    bytes   (b'' when unused)
cache_write_tokens int
```

Appending is backward compatible: `ByteHelper.read_bytes` at EOF reads a zero
length and returns `b''`, and `read_int` returns `0`, which is exactly how the
existing `completed` / `progress` fields already handle older peers. A node
running an older build ignores the tags: it will never store an entry, so the
origin's next attempt to reuse one simply misses.

New protocol number `CACHE_PROTOCOL = 3`, dispatched in
`ContentProvider._receive_data` (`src/language_pipes/content_provider/content_provider.py:150`)
alongside the existing `0` (job), `1` (RFM), `2` (cancel). Two tiny packets,
modeled on `JobCancel`:

- `CACHE_MISS(job_id, pipe_id)` — node → origin: "I could not adopt the prefix."
- `CACHE_ABORT(job_id, pipe_id)` — origin → nodes: "drop this job, I am restarting it."

---

## 6. Code changes by file

| File | Change |
|---|---|
| `jobs/prompt_cache.py` *(new)* | `CacheEntry`, `PromptCache`, block hashing, TTL/LRU sweep. |
| `jobs/cache_packets.py` *(new)* | `CacheMiss` / `CacheAbort` packets (mirrors `jobs/job_cancel.py`). |
| `util/oai_cache.py` *(new)* | Parse `prompt_cache_*` params and breakpoints; build usage details; validation errors. |
| `util/oai.py` | `ResponsesRequest` / `ChatCompletionRequest` carry a `CacheOptions`; usage blocks gain `input_tokens_details` / `prompt_tokens_details`. |
| `jobs/job.py` | New fields: `cache_options`, `cache_scope`, `cache_ids`, `cached_prefix_len`, `cache_write_points`, `cache_write_tokens`, `pending_write_id`. `past_seen_tokens()` adds the cached prefix. `to_network_job()` emits the tags. |
| `util/chunk_state.py` | `init(prompt_length, start_offset=0)` and offset-aware `get_range`. |
| `jobs/job_factory.py` | `start_job` accepts `cache_options` and stores it on the `Job`. |
| `jobs/job_processor.py` | `_state_embed`: plan the chain, adopt a local hit, init chunking with the offset. `_state_process_layers`: honor the write tag after computing. `_state_head`: at completion, tag/store the end-of-response boundary. |
| `jobs/job_tracker.py` | `add_job` adopts on `cache_use_id` or signals a miss; sweep the `PromptCache` in `check_stale_jobs`. |
| `jobs/job_receiver.py` | Send `CACHE_MISS`, handle `CACHE_ABORT`, restart a job with reuse disabled. |
| `content_provider/content_provider.py` | Own the `PromptCache`; dispatch protocol `3`. |
| `content_provider/job_provider.py` | `get/set_max_cache_time`; cache stats for the TUI. |
| `config.py` | `max_cache_time` field, default, save/load, `to_string()`. |
| `tui/.../jobs_server/top_state.py`, `tui/frame/tips.py` | New editable row + tip. |
| `documentation/{configuration,oai,architecture}.md` | Document the field, the API surface, and the fact that KV state now outlives a job. |

---

## 7. Failure modes

| Situation | Result |
|---|---|
| One node evicted its slice (different memory pressure) | `CACHE_MISS` → abort → full prefill. Correct, one wasted chunk. |
| A node runs an older build | Never stores; every reuse attempt misses; requests still succeed. |
| Pipe re-formed, a node now hosts a different layer range | Entry's `process_id`/layer range no longer match → treated as a miss. |
| Node dies mid-job | Unchanged from today: job expires after `EXPIRED_JOB_TIME`. Its entries die with it. |
| `restart_token` fires (hash validation failure) | Reuse for that job is disabled from that point; snapshot validation prevents a drifted node from storing a bad entry. |
| `max_cache_time = 0` on one node | That node never stores, so hits require the others to miss too — effectively disables reuse for pipes through it. Correct, just slower. |
| Two concurrent jobs on the same prefix | Both adopt independent copies; no interference. |
| Client changes one token in the middle of the prompt | Chain diverges at that block; everything before it still hits. |

The invariant that keeps all of this safe: **a node that cannot prove it holds
exactly the requested prefix refuses to compute the pass.** There is no path where
a partial or mismatched cache is silently used.

---

## 8. Privacy

`documentation/privacy.md` promises that only the end-model node sees text. Caching
does not weaken that, but it does add state that outlives a request, so the plan
takes three explicit positions:

1. **No cross-tenant reuse.** The chain is salted with the origin node ID and the
   API key, so an entry can only ever be reused by the same user from the same
   origin. This forgoes the "shared system prompt across all users" win on purpose.
2. **IDs carry no plaintext.** Layer nodes see 32-byte hashes salted with a value
   they do not know; they cannot test a guessed prompt against them.
3. **Bounded lifetime, no persistence.** Entries are in-memory only, capped by
   `max_cache_time`, and dropped on model unload, pipe teardown, and shutdown
   (hook into the existing `ModelManager` job hooks and `ContentProvider.stop_network`).

The docs should say plainly that with caching enabled a node retains derived KV
state for up to `max_cache_time` after a request finishes, and that setting it to
`0` restores the previous behavior.

---

## 9. Metrics and TUI

`PromptCache` tracks `hits`, `misses`, `entries`, `bytes`, `evictions`. The Jobs /
Server page shows one line under the new field:

```
   Cache: 6 entries, 1.2 GB, 74% hit rate
```

`Job` carries `cached_tokens` and `cache_write_tokens` so the active-jobs view can
show "prefill skipped: 3968 tokens" — the clearest signal that the feature is
working. Hit/miss counts also go into the existing per-job log line.

---

## 10. Testing

Unit (`tests/language_pipes/unit/`):

- `test_prompt_cache.py` — chain determinism; divergence at the first differing
  block; scope separation by API key / `prompt_cache_key` / origin; TTL expiry;
  LRU eviction; `adopt` returns an independent copy; entry invalidated by a
  changed `process_id` or layer range.
- `test_chunk_state.py` (extend) — offset init, ranges, `get_tokens_processed`,
  offset + a suffix shorter than `CHUNK_SIZE`.
- `test_job.py` (extend) — `past_seen_tokens` with a cached prefix, during prefill
  and after the first decode step.
- `test_network_job.py` (extend) — round-trip with tags; and a payload written
  without them (old peer) still parses with empty tags.
- `test_oai_responses.py` (extend) — parameter parsing and validation errors;
  breakpoint offset mapping including the "not a real prefix" rejection;
  `usage.input_tokens_details` shape in both streaming and non-streaming.
- `job_processor/` — a hit skips prefill chunks; write tags are emitted at the
  expected boundaries in each mode; a `CACHE_MISS` aborts and restarts once, with
  reuse disabled the second time.

Integration (`tests/language_pipes/integration/oai.py`): two requests sharing a
long prefix against a live pipe — assert the second reports `cached_tokens > 0`,
produces the same output distributionally, and has a lower time-to-first-token.
Worth running once against a hybrid-attention model (Qwen3.5) and a
sliding-window one (Gemma 3), since those are where snapshot-not-crop matters.

---

## 11. Phasing

**Phase 1 — single-node correctness.** `PromptCache`, chain IDs, `ChunkState`
offset, `past_seen_tokens`, adopt/store on the origin only (end-model layers),
`max_cache_time` config + TUI + docs. Reuse only when the origin hosts the whole
pipe locally. Ships a real feature and gets the resume-from-offset path proven
without any protocol change.

**Phase 2 — distributed reuse.** `NetworkJob` tags, store-on-tag in layer nodes,
`CACHE_MISS` / `CACHE_ABORT`, restart-with-reuse-disabled. This is the phase that
needs the most integration testing.

**Phase 3 — full OpenAI surface.** `prompt_cache_options`, explicit breakpoints,
breakpoint→offset mapping, `cache_write_tokens`, chat-completion
`stream_options.include_usage`.

**Phase 4 — optimizations.** Publish held-ID digests in DSN state to avoid doomed
attempts; per-block snapshots for partial system-prompt reuse; CPU demotion of
cold entries; wire the memory cap to the node memory limit.

---

## 12. Decisions worth a second opinion

1. **Scope isolation vs. hit rate.** Salting with the API key and origin node ID
   means a fleet of clients sharing one large system prompt gets no shared cache
   unless they share an API key. That is the right default for this project's
   privacy posture, but if a deployment wants it, an opt-in
   `shared_prompt_cache = true` on the jobs server could drop the API key from
   the salt. Not in this plan.
2. **`MIN_CACHE_TOKENS = 256`, not OpenAI's 1024.** Language Pipes runs much
   smaller models over much slower hops than OpenAI's fleet, so the prefill saved
   by a 256-token prefix is well worth an entry. It does mean `cached_tokens` can
   be non-zero here where OpenAI would report zero.
3. **Optimistic reuse over a prepare/ACK barrier.** Costs a wasted chunk on a
   miss; saves a round trip on every hit. Revisit if misses turn out to be common
   over WAN links.
4. **The `restart_token` duplicate-append hazard** (§4.3) is pre-existing and
   should be confirmed and fixed on its own, not folded into this work.
