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

Two new fields on the jobs server.

`src/language_pipes/config.py` (alongside `max_node_jobs` / `max_api_jobs`):

```toml
# Maximum seconds a cached prompt prefix is kept after its last use.
max_cache_time = 300

# Maximum tokens of KV state this node holds for the prompt cache, counting both
# stored entries and the reservations of jobs still in flight.
max_cache_tokens = 16384
```

- `DEFAULT_MAX_CACHE_TIME = 300` (5 minutes), seconds, int.
- `DEFAULT_MAX_CACHE_TOKENS = 16384`, tokens, int.
- **Either field at `0` disables prompt caching on this node**: no reads, no writes,
  no memory held.
- Both are read/written through `JobProvider.get_/set_max_cache_time` and
  `.../max_cache_tokens` (`src/language_pipes/content_provider/job_provider.py`),
  like the existing two limits.
- Effective TTL for a request = `min(requested_ttl_or_default, max_cache_time)`.
  A client asking for `"24h"` on a node configured with 300s gets 300s; this is not
  an error, since OpenAI treats retention as a request, not a contract.
- The lifetime is measured **from last use**, not from creation — reusing a prefix
  refreshes it, matching OpenAI ("a busy prefix stays warm, an idle one expires").
- Every node applies its own two limits to its own slice. Nodes are not required to
  agree; disagreement makes hits rarer (§7), never wrong.

### 3.1 Sizing `max_cache_tokens`

The budget is counted in **tokens, not bytes**, for two reasons: tokens are the unit
the API already reports (`cached_tokens`), and — the operative one — a token count is
knowable *before* the KV state exists, which is what admission control needs (§4.7).

Bytes per token depend on the model and on how much of it a node hosts:

```
bytes/token = 2 (K and V) x kv_heads x head_dim x dtype_size x layers_hosted_here
```

Qwen3-1.7B in bf16 (8 KV heads, head_dim 128) is 4 KB per token per layer. A node
hosting all 28 layers spends ~115 KB/token, so `max_cache_tokens = 16384` is ~1.8 GB;
a node hosting 4 layers spends ~16 KB/token, or ~270 MB for the same setting. Two
consequences worth documenting: the same number means very different memory on
different nodes, and a node hosting more layers should generally be given a *smaller*
token budget, not a larger one.

Size it against `max_api_jobs` too. Every in-flight job reserves
`prompt + max_response` tokens against the same budget (§4.7), so a node that allows
5 concurrent jobs with 4k prompts and 1k responses needs ~25k tokens of headroom
before a single entry can be *stored*, let alone kept.

TUI: two new editable rows on the Jobs / Server page
(`src/language_pipes/tui/components/jobs_server/top_state.py`), after "Max API Jobs".
`focus_idx` becomes: 0 port, 1 max node jobs, 2 max api jobs, 3 max cache time,
4 max cache tokens, 5 api keys, 6 start/stop — every `_on_enter` / `_on_prev` /
`_on_next` / `_get_tip_lines` / `get_footer` branch shifts accordingly. Two matching
`TIPS["jobs_server"]` entries in `src/language_pipes/tui/frame/tips.py`:

> Max Cache Time: How long (in seconds) a processed prompt prefix is kept in
> memory so a follow-up request that starts with the same text can skip
> re-processing it. Set to 0 to disable prompt caching.

> Max Cache Tokens: The total number of tokens this node will keep in the prompt
> cache, including tokens reserved by jobs that are still running. When a new job
> does not fit, the oldest cached prompts are dropped to make room. Set to 0 to
> disable prompt caching.

Docs to update: `documentation/configuration.md` (both fields under "API Server",
with the sizing formula), `documentation/oai.md` (a "Prompt Caching" section covering
§2), `documentation/architecture.md` (the KV-cache section currently says caches die
with the job).

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
    token_count: int         # i * BLOCK_SIZE, and this entry's charge on the budget
    cache: DynamicCache      # this node's slice only
    size_gb: float           # reporting only; the budget is counted in tokens
    created: float
    last_used: float
    expires_at: float        # last_used + min(requested_ttl, max_cache_time)

class PromptCache:
    def lookup(self, cache_id, model_id, process_id, start_layer, end_layer) -> Optional[CacheEntry]
    def adopt(self, entry) -> DynamicCache       # deep copy for the job to mutate
    def reserve(self, job_id, tokens) -> bool    # admission + eviction, see 4.7
    def release(self, job_id) -> None
    def store(self, cache_id, job, segment, token_count, ttl) -> None
    def used_tokens(self) -> int                 # entries + live reservations
    def sweep(self) -> None                      # TTL expiry
```

One `PromptCache` per node, owned by `ContentProvider` next to `JobTracker`, so both
the origin path (end model slice) and the layer path see the same store and the same
budget.

- **Adoption shares the K/V tensors; it does not copy them.** `DynamicLayer.update`
  and `DynamicSlidingWindowLayer.update` both do `self.keys = torch.cat([self.keys,
  new])` — they rebind the attribute to a freshly allocated tensor and never write
  into the old one (the docstring's "in-place" is wrong). So a job can share an
  entry's tensors and its first append leaves the entry untouched. `adopt` builds a
  new `DynamicCache` container pointing at the same tensors.
- **Recurrent layers are the exception and must be cloned.**
  `LinearAttentionLayer.update_recurrent_state` does
  `self.recurrent_states[i].copy_(new)` — a genuine in-place write into a
  static-address buffer, and the conv states work the same way. A shared recurrent
  state would be corrupted by the first pass of the borrowing job. These states are
  fixed-size (they do not grow with the prefix), so cloning them is cheap and does
  not scale with prefix length. `adopt` therefore clones any
  `LinearAttention*Layer` state and shares everything else.
- **The invariant needs a test, not just a comment**, since it depends on transformers
  internals: after a job adopts an entry and runs one pass, the entry's tensors must
  be unchanged in both content and `data_ptr`. A layer type that fails the check falls
  back to cloning. This is the one place where a transformers upgrade could silently
  corrupt cached state, so the test is the load-bearing part.
- **Eviction cannot disturb a running job.** Evicting drops the store's reference;
  Python's refcount keeps the tensors alive for whatever job is still using them.
- **Entries stay on the compute device.** Moving to CPU would save VRAM but give back
  much of the latency win on adopt. If VRAM pressure dominates, an entry can be
  demoted to CPU after one TTL period without changing anything else.
- **Two eviction triggers.** TTL expiry runs on the existing 10s
  `JobTracker.check_stale_jobs` cadence rather than adding a thread; budget eviction
  runs synchronously at admission (§4.7). Both call the same `gc.collect()` /
  `torch.cuda.empty_cache()` / `malloc_trim` sequence that job cleanup already uses.
- `size_gb` is measured with the same tensor walk as `Job.get_job_ram`
  (`src/language_pipes/jobs/job.py`) and exists only for the TUI; nothing enforces it.

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

**Stores do not copy either.** Because appends rebind rather than mutate, storing an
entry is just retaining a reference to the job's current per-layer tensors. The write
itself allocates nothing; what it costs is that the tensors stop being freed when the
job's cache grows past that boundary (see §4.7).

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
   Miss → the node does **not** process the packet; it replies `CacheStatus(MISS)`
   to the origin. A node that holds the prefix but has no budget for the job (§4.7)
   still adopts and computes; it replies `CacheStatus(NO_STORE)` so the origin stops
   tagging write points.
4. On `CacheStatus(MISS)` the origin broadcasts `CacheStatus(ABORT)` to every segment node
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

### 4.7 Admission control and the token budget

Each node tracks what the cache holds and what it has promised to hold:

```
used = sum(e.token_count for e in entries) + sum(r.tokens for r in live_reservations)
```

When a job starts, the node estimates what that job will ultimately cost the cache:

```
estimate = cached_prefix_len + prompt_tokens + max_completion_tokens
```

- `cached_prefix_len` — the entry being reused stays resident for the life of the job.
- `prompt_tokens + max_completion_tokens` — the job's own working cache, which is what
  gets stored at the write points.

The prefix is counted twice because it really is resident twice while the job runs, and
that is worth being precise about, since nothing in the design deliberately duplicates it:

- Adoption itself copies nothing (§4.2) — the job starts out sharing the entry's tensors.
- The job's first append calls `torch.cat`, which **allocates a new tensor of
  `prefix + chunk`** for each layer. That allocation happens whether or not the entry
  exists; it is how `DynamicCache` grows.
- The entry keeps its original `prefix` tensors alive. So from the first pass onward,
  memory holds `prefix` (entry) + `prefix + generated` (job).

The same doubling appears without any reuse at all: a job that stores an entry at its
prompt boundary and keeps decoding ends up holding both the stored tensors and its own
grown ones. Retaining a prefix costs one extra copy of it for as long as some job is
growing past it. That is inherent to a contiguous per-layer cache; the real fix is a
paged/block KV cache with shared blocks, which would need attention kernels that take
block tables and is well outside this plan.

`max_completion_tokens` is the ceiling the client asked for
(`Job.max_completion_tokens`), so the estimate is an upper bound — a job that stops
early releases more than it used.

Admission:

```
while used + estimate > max_cache_tokens and an evictable entry exists:
    evict the least recently used entry
if used + estimate > max_cache_tokens:
    run this job uncached   # no reuse, no store, no reservation
else:
    hold a reservation of `estimate` tokens for the life of the job
```

Reservations are released in `JobTracker.complete_job` and `remove_job`, and by the
stale-job sweep, so a dropped connection or an expired job cannot leak budget.

Points worth being explicit about:

- **Running uncached is not a rejection.** The job's own KV is ordinary inference
  memory, bounded by `max_node_jobs` / `max_api_jobs` as it is today. `max_cache_tokens`
  bounds only what the *cache* holds and what it is about to hold. A client that asks
  for `max_output_tokens: 128000` will fail admission on its own and run exactly as it
  does now, with a log line saying why.
- **Eviction cannot break a running job**, because adoption copies (§4.2). The
  least-recently-used entry can be dropped while a job is mid-decode against its copy.
- **Every node runs the same arithmetic against its own budget.** The origin computes
  the estimate right after tokenize (the first moment `prompt_tokens` is known) and
  puts it in `cache_reserve_tokens` on the first `NetworkJob`; each layer node reserves
  in `JobTracker.add_job`.
- **A node that declines says so.** It replies `CacheStatus(NO_STORE)`, and the origin
  stops tagging write points for that job. Otherwise the other nodes would store a
  prefix that one node is missing — memory spent on an entry set that is guaranteed to
  miss and force a restart later.
- Eviction order is by `last_used`, so it is consistent with the TTL rule that reuse
  refreshes an entry: the "oldest" entry is the one nothing has wanted for the longest.
- **Adopt-by-move, when the budget is tight.** If an entry has no other user and
  admission would otherwise refuse the job, the entry can be handed over and deleted
  instead of shared. After the job's first `torch.cat` the original tensors are freed,
  so the peak is `prompt + generated` rather than `prefix + prompt + generated`, and
  the estimate drops by `cached_prefix_len`. The cost is that a concurrent request for
  the same prefix misses until this job completes and re-stores at a longer boundary.
  Worth having as the fallback before giving up and running uncached (phase 4).

---

## 5. The cache protocol

Everything in §4 assumes the origin can get one fact it cannot compute locally:
**does every other node on the pipe still hold its slice of this prefix?** The origin
only sees its own store. That question, and the recovery when the answer is no, is
what the protocol is for.

It is deliberately small: five tag fields riding on the packet that already flows
through every node, plus one back-channel packet for the answers that have nowhere
to ride.

### 5.1 Tags on `NetworkJob`

`src/language_pipes/jobs/network_job.py` gains seven fields, appended at the end of
the serialization:

```
attempt                int     bumped when the origin restarts this job_id
cache_use_id           bytes   prefix the receiver should adopt (b'' = none)
cache_use_tokens       int     how many tokens that prefix covers
cache_tokens_expected  int     tokens the sender believes the receiver's cache holds
cache_write_id         bytes   snapshot this pass under this id (b'' = don't)
cache_write_tokens     int     token count that snapshot covers
cache_reserve_tokens   int     the job's budget estimate (0 = caching off)
```

These are enough for the **write** path on their own. The origin knows where the
boundaries are; the nodes do not (they never see tokens), so the boundary has to be
told to them — but it can ride the packet that already visits every node in the pass.
Each node stores after computing, exactly once, with no agreement needed between
them. There is no "write protocol" because the packet *is* the coordination.

The **read** path is a tag too (`cache_use_id`), and on a hit that is the whole
story: every node adopts and the job proceeds with no extra messages.

### 5.2 The `CacheStatus` packet

`CACHE_PROTOCOL = 3`, dispatched in `ContentProvider._receive_data`
(`src/language_pipes/content_provider/content_provider.py:150`) next to `0` (job),
`1` (RFM), `2` (cancel). One packet type, modeled on `jobs/job_cancel.py`:

```
CacheStatus(job_id, pipe_id, attempt, reason)
reason ∈ { MISS, NO_STORE, ABORT }
```

| Reason | Direction | Meaning | Origin's response |
|---|---|---|---|
| `MISS` | node → origin | "I do not hold `cache_use_id`; I did not compute this pass." | Abort the attempt and restart with reuse off. |
| `NO_STORE` | node → origin | "I computed fine, but I have no cache budget for this job." | Stop tagging write points for this job. |
| `ABORT` | origin → nodes | "Drop this job; attempt `n` is dead." | — |

The negatives need their own channel because a job packet only ever flows *forward*
(to the node hosting the next layers) or, at `HEAD`, back to the origin. A node that
cannot compute has nothing to forward, and a node that computed fine but cannot store
has no field in the outgoing packet to say so. `JobReceiver.restart_token` is the one
existing exception — it bounces a `NetworkJob` back to the origin — and it works
precisely because the node in question also cannot proceed. `MISS` could be expressed
that way; it is a separate packet because `NO_STORE` and `ABORT` cannot be, and one
small packet type is easier to reason about than two mechanisms.

### 5.3 The three exchanges

**Hit (the common case, zero extra messages).**

```
origin  tokenize → chain → local hit at 3968 → adopt → embed [3968:4000] → local layers
        └─ send NetworkJob{attempt:0, use_id:h[31], use_tokens:3968, expected:3968, reserve:E}
node1   add_job: lookup h[31] → hit → adopt → compute → forward (tags carried through)
node2   same → forward
origin  HEAD → sample → next pass, untagged unless it lands on a write point
```

**Miss on one node.**

```
node2   add_job: lookup h[31] → not held → does NOT compute
        └─ CacheStatus(job_id, pipe_id, attempt=0, MISS) → origin
origin  attempt := 1; drop adopted cache; cached_prefix_len := 0; reuse disabled
        └─ CacheStatus(..., attempt=0, ABORT) → every segment node
        └─ re-dispatch from EMBED with a fresh cache, past_seen_tokens = 0
node1   ABORT → remove_job + drop queued packets for the job
```

`ABORT` matters because node1 **already computed** the first chunk against the
adopted prefix: its cache holds 4000 tokens while the origin is about to start again
from 0. Without dropping that job, the next packet arrives with masks and
`cache_position` describing a 32-token sequence at offset 0 while the local cache has
4000 entries — a shape mismatch at best, silently wrong attention at worst.

**No budget on one node.**

```
node2   add_job: reserve(E) fails after eviction → adopt/compute normally
        └─ CacheStatus(job_id, pipe_id, attempt, NO_STORE) → origin
origin  clears this job's write points; no node stores anything for it
```

Without it, nodes 1 and 3 would store a prefix node 2 is missing: memory spent on an
entry set that is guaranteed to miss, plus a wasted attempt and restart on the next
request that tries it.

### 5.4 Races, and why `attempt` carries the correctness

Messages between two nodes are reliable (the router is HTTP), but nothing orders
messages sent over *different* connections. So the origin's `ABORT` to node1 can lose
the race against its own attempt-1 job packet to node1. If `ABORT` were the only
mechanism, node1 would apply attempt 1 to the polluted attempt-0 cache.

`attempt` closes that on the data path, which makes it the load-bearing part:

- `Job.attempt` starts at 0 and is bumped by the origin on every restart.
- `Job.receive_network_job` compares: `network_job.attempt > self.attempt` → discard
  the local job's cache and rebuild it from scratch before processing;
  `network_job.attempt < self.attempt` → drop the packet as stale.
- `CacheStatus` carries the attempt it refers to. `ABORT` for an attempt older than
  the node's current one is ignored. A `MISS` for an attempt the origin has already
  restarted is ignored too — otherwise one late miss would abort the retry that was
  sent because of it, and a job could ping-pong.

With that in place, `ABORT` is an **optimization, not a correctness requirement**: it
frees the stale caches immediately instead of leaving them until the next packet
rebuilds them or the 60s stale sweep reaps them.

`cache_tokens_expected` is the same idea applied one level down. The sender states
how many tokens it believes the receiver's cache holds; a node whose slice contains
at least one ordinary attention layer can check its own `get_seq_length` against it
and refuse the pass on a mismatch rather than computing garbage. It costs nothing and
turns a whole class of silent divergence into a detectable condition — including the
pre-existing `restart_token` hazard (§4.3), where a re-run pass appends duplicate KV
on nodes that already computed it. A node hosting only linear-attention layers cannot
answer that question and falls back to trusting `attempt`.

### 5.5 Backward compatibility

Appending fields is safe: `ByteHelper.read_bytes` at EOF reads a zero length and
returns `b''`, and `read_int` returns `0` — exactly how the existing `completed` /
`progress` fields already tolerate older peers. An older node reads `attempt = 0` and
empty tags, so it never stores and never adopts; the origin's next reuse attempt
misses, restarts once, and the request succeeds uncached. A newer node receiving an
old packet sees `attempt = 0`, which matches a job that has never been restarted.
Mixed-version pipes therefore degrade to today's behavior instead of breaking.

---

## 6. Code changes by file

| File | Change |
|---|---|
| `jobs/prompt_cache.py` *(new)* | `CacheEntry`, `PromptCache`, block hashing, token budget + reservations, TTL and LRU eviction. |
| `jobs/cache_packets.py` *(new)* | `CacheStatus` packet with `MISS` / `NO_STORE` / `ABORT` reasons (mirrors `jobs/job_cancel.py`). |
| `util/oai_cache.py` *(new)* | Parse `prompt_cache_*` params and breakpoints; build usage details; validation errors. |
| `util/oai.py` | `ResponsesRequest` / `ChatCompletionRequest` carry a `CacheOptions`; usage blocks gain `input_tokens_details` / `prompt_tokens_details`. |
| `jobs/job.py` | New fields: `attempt`, `cache_options`, `cache_scope`, `cache_ids`, `cached_prefix_len`, `cache_write_points`, `cache_write_tokens`, `pending_write_id`. `past_seen_tokens()` adds the cached prefix. `to_network_job()` emits the tags. `receive_network_job()` compares `attempt` and rebuilds or drops. |
| `util/chunk_state.py` | `init(prompt_length, start_offset=0)` and offset-aware `get_range`. |
| `jobs/job_factory.py` | `start_job` accepts `cache_options` and stores it on the `Job`. |
| `jobs/job_processor.py` | `_state_embed`: plan the chain, run admission, adopt a local hit, init chunking with the offset. `_state_process_layers`: honor the write tag after computing. `_state_head`: at completion, tag/store the end-of-response boundary and release the reservation. |
| `jobs/job_tracker.py` | `add_job` reserves budget and adopts on `cache_use_id`, or signals `MISS` / `NO_STORE`; `complete_job` / `remove_job` / the stale sweep release reservations; sweep the `PromptCache` in `check_stale_jobs`. |
| `jobs/job_receiver.py` | Send `CacheStatus`, handle `ABORT`, restart a job with reuse disabled, stop tagging on `NO_STORE`. |
| `content_provider/content_provider.py` | Own the `PromptCache`; dispatch protocol `3`. |
| `content_provider/job_provider.py` | `get/set_max_cache_time`; cache stats for the TUI. |
| `config.py` | `max_cache_time` and `max_cache_tokens` fields, defaults, save/load, `to_string()`. |
| `tui/.../jobs_server/top_state.py`, `tui/frame/tips.py` | Two new editable rows (focus indices shift), two tips, cache stats line. |
| `documentation/{configuration,oai,architecture}.md` | Document the field, the API surface, and the fact that KV state now outlives a job. |

---

## 7. Failure modes

| Situation | Result |
|---|---|
| One node evicted its slice (TTL, or budget pressure from its own jobs) | `MISS` → abort → full prefill. Correct, one wasted chunk. |
| A node runs an older build | Never stores; every reuse attempt misses; requests still succeed. |
| Pipe re-formed, a node now hosts a different layer range | Entry's `process_id`/layer range no longer match → treated as a miss. |
| Node dies mid-job | Unchanged from today: job expires after `EXPIRED_JOB_TIME`. Its entries die with it. |
| `restart_token` fires (hash validation failure) | Reuse for that job is disabled from that point; snapshot validation prevents a drifted node from storing a bad entry. |
| `max_cache_time = 0` or `max_cache_tokens = 0` on one node | That node never stores, so every reuse attempt through it misses — reuse is effectively off for pipes crossing it. Correct, just slower. |
| Two concurrent jobs on the same prefix | Both adopt independent copies; no interference. Each reserves its own budget. |
| Job's estimate alone exceeds `max_cache_tokens` | Runs uncached. Nothing is evicted to make room for something that will never fit. |
| One node's budget is full, the rest have room | That node replies `NO_STORE`; the origin stops tagging, so no partial entry set is left behind to force a future restart. |
| Budget thrash (every job evicts the last one) | Hit rate falls to zero; nothing is incorrect. The TUI's token-usage and eviction counters are what make this diagnosable. |
| Client cancels mid-generation | Reservation released by `remove_job` / the stale sweep; no budget leak. |
| `ABORT` overtaken by the retry's job packet | `attempt` on the packet rebuilds the node's cache; the late `ABORT` names an older attempt and is ignored. |
| `MISS` arrives after the origin already restarted | Ignored — it names a dead attempt, so the retry is not aborted by the miss that caused it. |
| Node's cache length disagrees with `cache_tokens_expected` | Node refuses the pass instead of computing against a drifted cache. |
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

`PromptCache` tracks `hits`, `misses`, `entries`, `tokens`, `reserved`, `bytes`,
`evictions`, and `no_store` refusals. The Jobs / Server page shows one line under the
new fields:

```
   Cache: 6 entries, 11.2k/16.4k tokens (3.1k reserved), 1.2 GB, 74% hit rate
```

Token usage against the budget is the number an operator actually tunes on, so it
leads; bytes follow because that is what runs the machine out of memory.

`Job` carries `cached_tokens` and `cache_write_tokens` so the active-jobs view can
show "prefill skipped: 3968 tokens" — the clearest signal that the feature is
working. Hit/miss counts also go into the existing per-job log line.

---

## 10. Testing

Unit (`tests/language_pipes/unit/`):

- `test_prompt_cache.py` — chain determinism; divergence at the first differing
  block; scope separation by API key / `prompt_cache_key` / origin; TTL expiry;
  `adopt` returns an independent copy; entry invalidated by a changed `process_id`
  or layer range.
- `test_prompt_cache_share.py` — the sharing invariant: after adopting an entry and
  running a pass, the entry's tensors are unchanged in content and `data_ptr` for
  `DynamicLayer` and `DynamicSlidingWindowLayer`; a `LinearAttentionLayer` entry is
  cloned and survives a borrowing job's `copy_()`; a store taken by reference is
  unaffected by the job's later appends. This test is what protects the design from a
  transformers upgrade that starts mutating in place.
- `test_prompt_cache_budget.py` — `used_tokens` counts entries plus reservations;
  admission evicts least-recently-used until the estimate fits; an estimate larger
  than the whole budget evicts nothing and returns "uncached"; reservations released
  on complete, cancel, and stale expiry; a reused entry moves to the back of the
  eviction order; `max_cache_tokens = 0` short-circuits every path.
- `test_chunk_state.py` (extend) — offset init, ranges, `get_tokens_processed`,
  offset + a suffix shorter than `CHUNK_SIZE`.
- `test_job.py` (extend) — `past_seen_tokens` with a cached prefix, during prefill
  and after the first decode step.
- `test_network_job.py` (extend) — round-trip with tags; a payload written without
  them (old peer) still parses with empty tags and `attempt = 0`.
- `test_job.py` (extend) — `receive_network_job` rebuilds the cache on a higher
  `attempt`, drops a packet with a lower one, and refuses a pass whose
  `cache_tokens_expected` disagrees with the local cache length.
- `test_oai_responses.py` (extend) — parameter parsing and validation errors;
  breakpoint offset mapping including the "not a real prefix" rejection;
  `usage.input_tokens_details` shape in both streaming and non-streaming.
- `job_processor/` — a hit skips prefill chunks; write tags are emitted at the
  expected boundaries in each mode; a `MISS` aborts and restarts once, with reuse
  disabled the second time; a `NO_STORE` leaves the job running with no write tags.
- `test_job_receiver.py` (extend) — the two orderings that matter: an `ABORT` that
  arrives after the retry's job packet is ignored, and a `MISS` naming a dead attempt
  does not abort the retry. Both should fail loudly if `attempt` handling regresses.
- `test_config.py` (extend) — both fields round-trip through save/load and default
  correctly when absent from an existing config file.

Integration (`tests/language_pipes/integration/oai.py`): two requests sharing a
long prefix against a live pipe — assert the second reports `cached_tokens > 0`,
produces the same output distributionally, and has a lower time-to-first-token.
Worth running once against a hybrid-attention model (Qwen3.5) and a
sliding-window one (Gemma 3), since those are where snapshot-not-crop matters.

---

## 11. Phasing

**Phase 1 — single-node correctness.** `PromptCache`, chain IDs, `ChunkState`
offset, `past_seen_tokens`, the token budget with admission and LRU eviction,
adopt/store on the origin only (end-model layers), both config fields + TUI + docs.
Reuse only when the origin hosts the whole pipe locally. Ships a real feature and
proves the resume-from-offset and budget paths without any protocol change.

**Phase 2 — distributed reuse.** `NetworkJob` tags including `attempt`,
`cache_tokens_expected` and `cache_reserve_tokens`; per-node admission; store-on-tag in
layer nodes; `CacheStatus` in all three flavors; restart-with-reuse-disabled. The
`attempt` field is worth landing first and on its own — it is a correctness fix for
job restarts that stands independently of caching. This phase needs the most
integration testing.

**Phase 3 — full OpenAI surface.** `prompt_cache_options`, explicit breakpoints,
breakpoint→offset mapping, `cache_write_tokens`, chat-completion
`stream_options.include_usage`.

**Phase 4 — optimizations.** Publish held-ID digests in DSN state to avoid doomed
attempts; per-block snapshots for partial system-prompt reuse; CPU demotion of cold
entries; derive a default `max_cache_tokens` from the node memory limit and the
hosted layer count instead of a fixed number.

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
4. **"Oldest" is read as least-recently-used.** Eviction drops the entry nothing has
   touched for the longest, which keeps it consistent with the TTL rule that reuse
   refreshes an entry. Strict creation order would evict a hot long-lived system
   prompt in favor of a cold recent one. Say so if creation order was meant.
5. **The budget is per node, in tokens.** Tokens are knowable before the KV state
   exists, which is what admission needs, but they mean different amounts of memory
   on different nodes (§3.1). A byte-denominated limit would be more honest about
   memory and useless for admission; deriving the token budget from a byte limit
   plus the hosted layer count (phase 4) gets both.
6. **The `restart_token` duplicate-append hazard** (§4.3) is pre-existing and
   should be confirmed and fixed on its own, not folded into this work.
