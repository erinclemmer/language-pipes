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
hosts, and that cache is created in `Job.__init__` (`src/language_pipes/jobs/job.py:118`)
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
| `prompt_cache_key` | string | Scopes the cache. Requests with the same key + same prefix share entries. Optional; defaults to `""`. On a server with no `api_keys` configured it is also what separates one caller from another, so an absent key disables caching for the request (§2.3). |
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

### 2.3 Unauthenticated servers

`api_keys` defaults to empty (`src/language_pipes/config.py:164`), and `do_POST`
then skips authorization and passes the literal string `"anon"` down as the key
(`src/language_pipes/oai_server.py:50-53`). Every unauthenticated caller therefore
lands in the *same* cache scope, and since `prompt_cache_key` defaults to `""` the
default scope on such a node is one fixed value shared by everyone who can reach
the port. That is a cross-tenant `cached_tokens` oracle: a caller can confirm
another caller's prompt (and, because the completion boundary is also written, their
response) 128 tokens at a time by guessing it and reading `cached_tokens`.

So: **when `api_keys` is empty and `prompt_cache_key` is absent or `""`, the
request runs uncached** — no lookup, no store, no reservation. `cached_tokens`
is `0` and everything else about the request is unchanged.

This is a silent disable, not an error. Requiring the parameter would 400 every
plain chat request from a client that sends no cache options at all, including the
two-node example in the README, so a missing key means "no caching" the same way
`max_cache_time = 0` does. A request that *does* carry a key gets normal caching,
scoped to that key.

The honest framing of what this buys: with `api_keys` set, isolation rests on the
API key and `prompt_cache_key` is what OpenAI says it is — a partitioning label.
With `api_keys` empty, `prompt_cache_key` is doing the isolating, and it is only as
good as the value the client picks. OpenAI's own guidance produces low-entropy,
structured labels (`support-v3:user_123`), which are guessable; a caller who wants
isolation on an unauthenticated node needs an unguessable value, and one that is
sent in cleartext, since `OAIHttpServer` is a plain `ThreadingHTTPServer` with no
TLS. It defends against someone who can reach the API, not against someone who can
observe it. `documentation/oai.md` should say exactly that, and should say that
configuring `api_keys` is the supported way to get cache isolation.

Two consequences for the rest of the plan: the scope value is never logged (§9 logs
a truncated hash instead), and `"anon"` is never treated as an identity anywhere
else.

### 2.4 Response fields

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
§2, including the unauthenticated-server rule in §2.3 and the advice that `api_keys`
is the supported way to get cache isolation), `documentation/architecture.md` (the
KV-cache section currently says caches die with the job), and
`documentation/privacy.md` (that a node now retains derived KV state for up to
`max_cache_time` past the request, and what scopes it).

---

## 4. Core mechanism

### 4.1 Block-chained prefix IDs

Prefix identity is a hash chain over fixed-size token blocks, so that two
requests that share the first K blocks produce the same first K chain values.

The chain is **keyed**, under a secret only the origin holds:

```python
BLOCK_SIZE = 128          # tokens; must be a multiple of CHUNK_SIZE (32)
MIN_CACHE_TOKENS = 256    # 2 blocks; shorter prefixes are never cached

# Per origin, per process. Generated at ContentProvider startup, never sent
# over the network, never written to disk, never logged.
_CACHE_SECRET = secrets.token_bytes(32)

def _scope(origin_node_id: str, api_key: str, prompt_cache_key: str) -> bytes:
    return hmac.new(
        _CACHE_SECRET,
        b"lp-prompt-cache-v1|scope"
        + sha256(origin_node_id.encode()).digest()
        + sha256(api_key.encode()).digest()
        + sha256(prompt_cache_key.encode()).digest(),
        sha256,
    ).digest()

def _link(prev: bytes, block: List[int]) -> bytes:
    return hmac.new(
        _CACHE_SECRET,
        b"lp-prompt-cache-v1|blk" + prev
        + np.asarray(block, dtype="<i4").tobytes(),
        sha256,
    ).digest()

h[0] = _scope(origin_node_id, api_key, prompt_cache_key)
h[i] = _link(h[i-1], tokens[(i-1)*BLOCK_SIZE : i*BLOCK_SIZE])
```

`h[i]` is the 32-byte ID of the prefix that is exactly `i * BLOCK_SIZE` tokens long.

Two properties of this construction are load-bearing, and both were arrived at by
asking what a *layer node* can do with the IDs it sees on the wire:

- **Every link is keyed, not just the root.** The intermediate `h[i]` are not
  private — they ride the packet as `cache_use_id` / `cache_write_id` (§5.1), so
  every node on the pipe observes them. If only `h[0]` were keyed and the links were
  plain `sha256`, a node holding an observed `h[31]` could compute
  `sha256(h[31] + guess)` offline and test it against the next `cache_write_id` it
  sees. The oracle would survive, just starting one block further in. Keying each
  link means an observed ID cannot be extended, only compared.
- **Fields are digested before concatenation**, so the boundaries between them are
  unambiguous. `prompt_cache_key` is arbitrary client text; with plain
  concatenation, `api_key="ops"` + `prompt_cache_key="-readonly:x"` and
  `api_key="ops-readonly"` + `prompt_cache_key=":x"` produce the *same* scope, and
  one tenant reads another's cache. Fixed-width digests make that impossible for any
  choice of either field. (Keys minted by the TUI are `secrets.token_urlsafe(32)`
  and are not prefix-related, but hand-typed keys and TOML-edited `api_keys` have no
  such property.)

The rest follows as before:

- **The chain is computed only on the origin**, because only the origin ever sees
  tokens (`EndModel.tokenize`). Layer nodes receive opaque 32-byte IDs. With the
  chain keyed under a secret the origin never transmits, those IDs are not merely
  opaque but *unverifiable*: a node cannot test a guessed prompt against them at
  any offset, even knowing `origin_node_id` (which is in every `NetworkJob`,
  `src/language_pipes/jobs/network_job.py:49`), the API key, and the cache key.
  This is the difference between "the attacker must invert hidden states" —
  expensive, weight-dependent, per-request, and the bound
  `documentation/privacy.md` already reasons about — and "the attacker can confirm
  a guess with one hash", which caching would otherwise hand them for free.
- The scope includes `origin_node_id`, the API key and `prompt_cache_key`, so
  **entries are never shared across users or across origin machines**. This is a
  privacy decision, not a performance one: a cross-tenant hit is an oracle that
  tells one user another user sent a particular prefix. It also means a shared
  layer node cannot correlate two different users' prompts (§8).
- `prompt_cache_key` further partitions within one API key, which is what clients
  use to keep unrelated workloads from evicting each other — and on an
  unauthenticated node it is the only thing partitioning at all (§2.3).
- Block size 128 gives `cached_tokens` the same 128-granularity OpenAI reports,
  and divides evenly into `CHUNK_SIZE = 32` so cache boundaries always fall on
  prefill chunk boundaries.

Cost is negligible: ~32 HMACs over ~512 bytes each for a 4k prompt, on the origin,
once per request. Nothing else in the design needs to recompute a chain value —
§4.4 no longer publishes IDs anywhere, so the origin is the only producer and every
other party is a comparator.

One behavioral consequence to be aware of: **the secret is per process, so entries
do not survive an origin restart.** The origin's own entries died with the process
anyway; what changes is that layer nodes' slices for that origin become unreachable
and idle out over `max_cache_time` instead of hitting. This is consistent with §7
already treating a changed `process_id` as a miss, and it is the conservative
direction — a restarted origin cannot adopt state it can no longer name.

### 4.2 Node-local store

New file `src/language_pipes/jobs/prompt_cache.py`:

```python
@dataclass
class CacheEntry:
    cache_id: bytes          # h[i]
    origin_node_id: str      # the only node allowed to reuse this entry
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
    def lookup(self, cache_id, origin_node_id, model_id, process_id, start_layer, end_layer) -> Optional[CacheEntry]
    def adopt(self, entry) -> DynamicCache       # deep copy for the job to mutate
    def reserve(self, job_id, tokens) -> bool    # admission + eviction, see 4.7
    def release(self, job_id) -> None
    def store(self, cache_id, origin_node_id, job, segment, token_count, ttl) -> None
    def used_tokens(self) -> int                 # entries + live reservations
    def sweep(self) -> None                      # TTL expiry
```

One `PromptCache` per node, owned by `ContentProvider` next to `JobTracker`, so both
the origin path (end model slice) and the layer path see the same store and the same
budget.

- **An entry is bound to its origin, and `lookup` enforces it.** A layer node's store
  holds entries for every origin it serves, so `cache_id` alone must not be what
  authorizes adoption — it is a value that every node on the pipe *observes* in the
  packet tags, and a bare `cache_id` check would make it a bearer token. `store`
  records `network_job.origin_node_id`; `lookup` requires
  `entry.origin_node_id == network_job.origin_node_id` alongside the existing
  `model_id` / `process_id` / layer-range checks, and treats a mismatch as a miss.
  The sender's node identity is already authenticated by the DSN transport, so this
  costs nothing and is not spoofable by a peer.

  Without it, a node that observed an ID on a pipe it participates in could replay
  that ID to a *different* node — one hosting a layer range it does not itself host
  — and have that node compute the attacker's own suffix on top of the victim's
  adopted KV state. The returned hidden states encode the victim's prompt at a depth
  the attacker was never given, which `documentation/privacy.md` treats as
  invertible given the public weights. The keyed chain (§4.1) stops an ID from being
  *derived*; this binding stops an observed one from being *reused*. Both are needed:
  the network is open to any authenticating peer unless `whitelist_node_ids` is set
  (`src/language_pipes/config.py:265`), and any peer can drive job packets into a
  node (`content_provider/content_provider.py:150`).

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
On mismatch it skips the store (and logs at debug). This is cheap insurance against a
node whose cache has drifted. The known way that happened — the `restart_token` bug
fixed in §11 phase 0 — is closed by the `pass_idx` replay mechanism, and a node that
receives a pass out of sequence refuses it (§5.4), so drift should no longer be
reachable; the check stays because a drifted node must never be allowed to write its
slice into an entry that later requests will adopt.

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
   `job.caching.prefix_len`, and tags the first outgoing `NetworkJob` with
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

**Not doing: publishing held-ID digests.** An earlier draft had each node advertise
the set of IDs it holds in DSN shared state so the origin could skip a doomed
attempt. That is dropped deliberately. An ID is not just an identifier here — until
`lookup` checks the origin binding (§4.2) it is the thing that names reusable KV
state, and even with the binding it is a value that should stay confined to the
pipes that legitimately carry it. Broadcasting the full set of live IDs to every
peer in the swarm hands an attacker the input they cannot otherwise obtain, and
does it for nodes they never shared a pipe with.

The protocol already covers the case: `CacheStatus(MISS)` tells the origin, and
§4.4 bounds the waste at one 32-token chunk through part of the pipe on a miss —
a cost paid only on the uncommon path, while the hit path was already zero extra
messages. Letting a job get partway down the pipe to discover a miss is the right
trade against publishing a swarm-wide index of cache state. If the miss rate ever
justifies revisiting this, the shape to consider is a per-origin membership hint
served only to that origin, not a shared digest.

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

- `Job.past_seen_tokens()` (`src/language_pipes/jobs/job.py:138`) becomes
  `self.caching.prefix_len + self.chunking.get_tokens_processed()` during prefill.
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
estimate = caching.prefix_len + prompt_tokens + max_completion_tokens
```

- `caching.prefix_len` — the entry being reused stays resident for the life of the job.
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
  the estimate drops by `caching.prefix_len`. The cost is that a concurrent request for
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
the serialization. `pass_idx` landed first, with the phase 0 restart fix (§11), and is
currently the last field on the wire; the other six arrive with phase 2, appended after
it:

```
pass_idx               int     sequence number of this pass within the job (phase 0,
                               numbered from 1; 0 = peer predates the field)
attempt                int     bumped when the origin restarts this job_id from scratch
cache_use_id           bytes   prefix the receiver should adopt (b'' = none)
cache_use_tokens       int     how many tokens that prefix covers
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
existing exception — it bounces a `NetworkJob` back to the origin, which after phase 0
replays the pass (§11) — and it works precisely because the node in question also
cannot proceed. `MISS` could be expressed that way; it is a separate packet because
`NO_STORE` and `ABORT` cannot be, and one small packet type is easier to reason about
than two mechanisms. The two restarts are also different operations: a bounce is a
*replay* of the same pass against unchanged caches, a `MISS` is a *rebuild* from token
0 against fresh ones.

### 5.3 The three exchanges

**Hit (the common case, zero extra messages).**

```
origin  tokenize → chain → local hit at 3968 → adopt → embed [3968:4000] → local layers
        └─ send NetworkJob{pass:0, attempt:0, use_id:h[31], use_tokens:3968, reserve:E}
node1   add_job: lookup h[31] → hit → adopt → compute → forward (tags carried through)
node2   same → forward
origin  HEAD → sample → next pass, untagged unless it lands on a write point
```

**Miss on one node.**

```
node2   add_job: lookup h[31] → not held → does NOT compute
        └─ CacheStatus(job_id, pipe_id, attempt=0, MISS) → origin
origin  attempt := 1; drop adopted cache; caching.prefix_len := 0; reuse disabled
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

- `attempt` starts at 0 and is bumped by the origin on every restart. It lives on
  `PassSequence` (`jobs/pass_sequence.py`) beside `pass_idx`.
- `Job.receive_network_job` compares: `network_job.attempt > passes.attempt` → discard
  the local job's cache and every saved pass payload, `passes.reset()`, and rebuild
  from scratch before processing; `network_job.attempt < passes.attempt` → drop the
  packet as stale.
- `CacheStatus` carries the attempt it refers to. `ABORT` for an attempt older than
  the node's current one is ignored. A `MISS` for an attempt the origin has already
  restarted is ignored too — otherwise one late miss would abort the retry that was
  sent because of it, and a job could ping-pong.

With that in place, `ABORT` is an **optimization, not a correctness requirement**: it
frees the stale caches immediately instead of leaving them until the next packet
rebuilds them or the 60s stale sweep reaps them.

`pass_idx` (§11 phase 0, already shipped) is the same idea applied one level down,
within an attempt. The origin numbers every pass it dispatches, from 1; a node keeps
what it forwarded for each entry point it was visited at, tagged with that number, and
on the next packet either forwards the saved payload again (same number, same entry
point — the cache is not touched) or computes (`last + 1`, or a second visit of the
same pass at a different entry point). Anything else means this node's cache cannot be
right for the incoming pass, and it refuses rather than computing garbage. `0` means
the peer predates the field, and the check is skipped. On a rebuild the origin resets
the whole `PassSequence` along with the bump to `attempt`, so the pair
`(attempt, pass_idx)` totally orders every pass a job has ever sent. This is exact
where a `get_seq_length` comparison would be inferred, and it works on a slice made
only of linear-attention layers, which cannot report a sequence length at all.

Put `attempt` on `PassSequence` next to `pass_idx` when phase 2 lands it, and give the
class a `reset()`: the two numbers are one ordering and should not drift apart across
two objects. Keep the phase 0 split — `PassSequence` decides, `Job` applies the
decision to its own fields.

The five cache tags are the same story one object over: they belong on `JobCache`
(`jobs/job_cache.py`, reached as `Job.caching`), which already holds the chain ids and
write points they carry. Phase 1 put the whole read/write path there, so phase 2 adds
`cache_use_id` / `cache_use_tokens` and the reservation estimate to that object rather
than to `Job`, and `receive_network_job` copies the write tag into the same
`pending_write_id` / `pending_write_tokens` the origin already sets on itself.

### 5.5 Backward compatibility

Appending fields is safe: `ByteHelper.read_bytes` at EOF reads a zero length and
returns `b''`, and `read_int` returns `0` — exactly how the existing `completed` /
`progress` fields already tolerate older peers. An older node reads `attempt = 0` and
empty tags, so it never stores and never adopts; the origin's next reuse attempt
misses, restarts once, and the request succeeds uncached. A newer node receiving an
old packet sees `attempt = 0`, which matches a job that has never been restarted, and
`pass_idx = 0` on every pass. Because the origin numbers from 1, `0` is unambiguous:
it means "peer does not number passes", and the sequence check is skipped for that
packet, which is today's behavior.
Mixed-version pipes therefore degrade to today's behavior instead of breaking.

---

## 6. Code changes by file

| File | Change |
|---|---|
| `jobs/prompt_cache.py` *(new)* | `CacheEntry`, `PromptCache`, the keyed block chain and its per-process secret, origin binding on `store`/`lookup`, token budget + reservations, TTL and LRU eviction. |
| `jobs/cache_packets.py` *(new)* | `CacheStatus` packet with `MISS` / `NO_STORE` / `ABORT` reasons (mirrors `jobs/job_cancel.py`). |
| `util/oai_cache.py` *(new)* | Parse `prompt_cache_*` params and breakpoints; build usage details; validation errors; disable caching when the server is unauthenticated and no `prompt_cache_key` was sent (§2.3). |
| `util/oai.py` | `ResponsesRequest` / `ChatCompletionRequest` carry a `CacheOptions`; usage blocks gain `input_tokens_details` / `prompt_tokens_details`. |
| `jobs/network_job.py` | `pass_idx` (phase 0, ✅ landed as the last appended field), then `attempt` and the five cache tags (phase 2), appended after it in that order. |
| `jobs/pass_sequence.py` *(new, phase 0)* ✅ | `PassSequence`, `SavedPass`, `MAX_PASS_RETRIES`. Owns the pass numbering, the payload each entry point last forwarded, the sequence check, the bounce answer and the retry cap. Phase 2 adds `attempt` and `reset()` here. |
| `jobs/job_cache.py` *(new, phase 1)* ✅ | `JobCache`, reached as `Job.caching`. Owns everything one job knows about the prompt cache: the request's `CacheOptions`, the `scope`, the chain `ids`, the adopted `prefix_len`, the reported `cached_tokens`, the `write_points`, the pending write tag and the reservation flag; plus the decisions that read only those — `searched()`, `forget()`, `adopt()`, `plan_prompt_write()`, `next_write_point()`, `tag()` / `take_pending()`, `response_write_point()` and `log_fields()`. Phase 2 adds the incoming `cache_use_*` tags and the reservation estimate here. Same split as `PassSequence`: this object decides, `Job` and the processor apply. |
| `jobs/job.py` | Phase 0 ✅: `passes: PassSequence`, `replay(saved)`, and the two branches in `receive_network_job()` that ask it what to do. Phase 1 ✅: `caching: JobCache` — one field, not nine — and `past_seen_tokens()` / `init_chunking()` reading `caching.prefix_len`. `Job.cache` keeps its old meaning: the working `DynamicCache` for the layers this node hosts. Phase 2: `to_network_job()` emits the tags from `caching`, `receive_network_job()` compares `attempt` and rebuilds or drops. |
| `util/chunk_state.py` | `init(prompt_length, start_offset=0)` and offset-aware `get_range`. |
| `jobs/job_factory.py` | `start_job` accepts `cache_options` and stores it on `Job.caching`, then derives `caching.scope` there so the API key travels no further. |
| `jobs/job_processor.py` | Phase 0 ✅: `_state_embed` numbers the pass, `_state_send` saves what went out, `replaying` short-circuits to `SEND`. Phase 1/2: `_state_embed`: plan the chain, run admission, adopt a local hit, init chunking with the offset. `_state_process_layers`: honor the write tag after computing. `_state_head`: at completion, tag/store the end-of-response boundary and release the reservation. |
| `jobs/job_tracker.py` | `add_job` reserves budget and adopts on `cache_use_id` (passing `network_job.origin_node_id` to `lookup`), or signals `MISS` / `NO_STORE`; `complete_job` / `remove_job` / the stale sweep release reservations; sweep the `PromptCache` in `check_stale_jobs`. |
| `jobs/job_receiver.py` | Phase 0 ✅: `_process_network_job` split out of the runner loop; a refused pass becomes a `cancel_job` carrying `PassSequence.error`. Phase 2: send `CacheStatus`, handle `ABORT`, rebuild a job with reuse disabled, stop tagging on `NO_STORE`. |
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
| `restart_token` fires (hash validation failure) | The origin replays its saved output for the pass; nodes that already computed it replay theirs, nodes that did not compute it now. No cache is touched, so no drift, and cache reuse for the job is unaffected. After the per-pass retry cap the job is canceled with a reason. |
| `max_cache_time = 0` or `max_cache_tokens = 0` on one node | That node never stores, so every reuse attempt through it misses — reuse is effectively off for pipes crossing it. Correct, just slower. |
| Two concurrent jobs on the same prefix | Both adopt independent copies; no interference. Each reserves its own budget. |
| Job's estimate alone exceeds `max_cache_tokens` | Runs uncached. Nothing is evicted to make room for something that will never fit. |
| One node's budget is full, the rest have room | That node replies `NO_STORE`; the origin stops tagging, so no partial entry set is left behind to force a future restart. |
| Budget thrash (every job evicts the last one) | Hit rate falls to zero; nothing is incorrect. The TUI's token-usage and eviction counters are what make this diagnosable. |
| Client cancels mid-generation | Reservation released by `remove_job` / the stale sweep; no budget leak. |
| `ABORT` overtaken by the retry's job packet | `attempt` on the packet rebuilds the node's cache; the late `ABORT` names an older attempt and is ignored. |
| `MISS` arrives after the origin already restarted | Ignored — it names a dead attempt, so the retry is not aborted by the miss that caused it. |
| Packet's `pass_idx` is neither the node's last pass nor the next one | Node refuses the pass and cancels the job with a reason instead of computing against a cache that cannot match. |
| Client changes one token in the middle of the prompt | Chain diverges at that block; everything before it still hits. |
| Origin restarts | New per-process cache secret, so every prior ID is unnameable. Layer nodes' orphaned slices idle out over `max_cache_time`. Correct, one cold period. |
| Unauthenticated node, request with no `prompt_cache_key` | Runs uncached; `cached_tokens` is 0 (§2.3). |
| An ID observed on the pipe is replayed by another node | `lookup` requires the entry's `origin_node_id` to match the requesting job's; mismatch is a miss (§4.2). |

The invariant that keeps all of this safe: **a node that cannot prove it holds
exactly the requested prefix refuses to compute the pass.** There is no path where
a partial or mismatched cache is silently used.

---

## 8. Privacy

`documentation/privacy.md` promises that only the end-model node sees text. Caching
does not weaken that, but it does add state that outlives a request, so the plan
takes three explicit positions:

1. **No cross-tenant reuse.** The chain is keyed over the origin node ID, the API
   key and `prompt_cache_key`, each digested before it is combined (§4.1), so an
   entry can only ever be reused by the same user from the same origin — and no
   choice of the client-controlled `prompt_cache_key` can collide into another
   tenant's scope. This forgoes the "shared system prompt across all users" win on
   purpose. On a node with no `api_keys`, where every caller is `"anon"`, the
   guarantee rests on `prompt_cache_key` alone, so a request without one is not
   cached at all (§2.3).
2. **IDs are unguessable and unverifiable.** Layer nodes see 32-byte values from an
   HMAC chain under a per-origin secret that is never transmitted. They cannot test
   a guessed prompt against an observed ID, and cannot extend one to predict the
   next, even knowing every other input to the chain. This matters because
   `origin_node_id` travels in the clear in every `NetworkJob` and the API key may
   be a constant: an unkeyed hash would have made prompt confirmation a cheap
   offline operation for any node on the pipe, which is a strictly stronger
   capability than the hidden-state inversion `documentation/privacy.md` already
   treats as the layer node's ceiling.
3. **An ID is not a capability.** Adoption additionally requires that the entry was
   created by the same origin that is now asking for it (§4.2), so an observed ID
   cannot be replayed to a node hosting layers the observer does not host. Held IDs
   are never published into DSN shared state (§4.4).
4. **Bounded lifetime, no persistence.** Entries are in-memory only, capped by
   `max_cache_time`, and dropped on model unload, pipe teardown, and shutdown
   (hook into the existing `ModelManager` job hooks and `ContentProvider.stop_network`).
   The per-process secret means a restart also makes every prior entry unnameable.

The docs should say plainly that with caching enabled a node retains derived KV
state for up to `max_cache_time` after a request finishes, and that setting it to
`0` restores the previous behavior.

Two limits worth stating rather than implying. The scope value and
`prompt_cache_key` are never logged — §9's per-job line carries a truncated hash of
the scope, not the key. And none of this is a defense against an observer of the
API traffic itself: the OAI server is a plain `ThreadingHTTPServer` with no TLS
(`src/language_pipes/oai_server.py:107`), so `prompt_cache_key`, like the API key
beside it, is in cleartext on the wire. `documentation/privacy.md` Attack Vector 4
already tells operators to firewall or bind the API locally; the cache scoping story
assumes they did.

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

`Job.caching` carries `cached_tokens` and `cache_write_tokens` so the active-jobs view can
show "prefill skipped: 3968 tokens" — the clearest signal that the feature is
working. Hit/miss counts also go into the existing per-job log line, identifying the
scope by the first 8 hex characters of `h[0]`. Neither `prompt_cache_key` nor the
API key is ever logged: on an unauthenticated node the cache key is what separates
one caller from another (§2.3), so putting it in a log file would undo that.

---

## 10. Testing

Unit (`tests/language_pipes/unit/`):

- `test_prompt_cache.py` — chain determinism *within one secret*; divergence at the
  first differing block; scope separation by API key / `prompt_cache_key` / origin;
  TTL expiry; `adopt` returns an independent copy; entry invalidated by a changed
  `process_id` or layer range. Plus three that exist because of the threat model:
  a fresh secret yields different IDs for the same tokens; `api_key="a"` +
  `prompt_cache_key="bc"` and `api_key="ab"` + `prompt_cache_key="c"` produce
  *different* scopes (the field-framing regression); and `lookup` with a
  non-matching `origin_node_id` returns `None` for an entry that matches on every
  other field.
- `test_prompt_cache_anon.py` — with `api_keys` empty: a request with no
  `prompt_cache_key` neither reads nor writes and reports `cached_tokens: 0`, a
  request with one caches normally, and two different keys do not see each other's
  entries. Same assertions with `api_keys` set confirm the absent-key case still
  caches there.
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
  and after the first decode step; `JobCache.next_write_point` through `job.caching`.
- `test_network_job.py` (extend) — round-trip with tags; a payload written without
  them (old peer) still parses with empty tags, `attempt = 0` and `pass_idx = 0`.
- ✅ `test_job.py` (phase 0, `JobReplayTests`) — a repeat of the last pass replays the
  saved output without calling the model and leaves the cache object and its length
  unchanged; `last + 1` computes; a second visit at another entry point computes; a
  first packet on a node that just joined is accepted; any other value refuses; a
  constant `0` from an old peer is accepted without the check.
- `test_job.py` (extend, phase 2) — `receive_network_job` rebuilds the cache and
  resets the whole `PassSequence` on a higher `attempt`, and drops a packet with a
  lower one.
- `test_oai_responses.py` (extend) — parameter parsing and validation errors;
  breakpoint offset mapping including the "not a real prefix" rejection;
  `usage.input_tokens_details` shape in both streaming and non-streaming.
- `job_processor/` — a hit skips prefill chunks; write tags are emitted at the
  expected boundaries in each mode; a `MISS` aborts and restarts once, with reuse
  disabled the second time; a `NO_STORE` leaves the job running with no write tags.
- ✅ `test_job_receiver.py` (phase 0, `PassSequenceTests`) — a refused pass cancels the
  job and notifies the origin; the retry cap cancels with its reason; a bounce for a
  pass older than the one in flight is dropped. The resend-without-embedding half lives
  in `job_processor/test_state_embed.py`, which can drive the whole origin FSM.
- `test_job_receiver.py` (extend, phase 2) — the two orderings that matter: an
  `ABORT` that arrives after the retry's job packet is ignored, and a `MISS` naming a
  dead attempt does not abort the retry. Both should fail loudly if `attempt` handling
  regresses.
- ✅ `job_processor/` (phase 0) — `test_state_layers.py`: a layer node replaying a pass
  forwards the saved `JobData` (state and `shared_kv_states`) and never calls
  `process_job`. `test_state_embed.py`: a prefill bounce at chunk *k* re-sends chunk
  *k*, not *k+1*, with `compute_embed` never called; a decode bounce leaves
  `current_token` and `input_ids` alone. `test_state_send.py`: the pass number goes on
  the wire and the payload is saved under its entry point. `test_restart.py`: two real
  `Job`s over the real wire format, asserting each node's cache holds every position
  exactly once across four corruption scenarios.
- `test_config.py` (extend) — both fields round-trip through save/load and default
  correctly when absent from an existing config file.

Integration (`tests/language_pipes/integration/oai.py`): two requests sharing a
long prefix against a live pipe — assert the second reports `cached_tokens > 0`,
produces the same output distributionally, and has a lower time-to-first-token.
Worth running once against a hybrid-attention model (Qwen3.5) and a
sliding-window one (Gemma 3), since those are where snapshot-not-crop matters.

---

## 11. Phasing

**Phase 0 — fix job restart, independent of caching. Shipped (`f049727`, `dfff22b`).** `JobReceiver.restart_token`
bounced a job back to the origin when a `NetworkJob`'s hash failed to validate, and the
origin resumed without telling the nodes that already computed the failed pass. Two live
defects followed, neither covered by a test:

- **Prefill:** the origin re-entered `_state_embed`, `chunking.is_active()` was true, so
  it called `chunking.advance()` unconditionally. The failed chunk was skipped rather
  than retried. `past_seen_tokens()` then counted it as processed while the nodes past
  the corruption point never computed it.
- **Decode:** the origin re-embedded the current token and re-ran its local layers, and
  every node upstream of the corruption point ran that token a second time. Because
  `DynamicLayer.update` concatenates, those nodes got a duplicate KV entry for one
  position while the downstream nodes had none.

Either way the pipe's caches disagreed about their own length, and the failure surfaced
later as a mask/shape mismatch or wrong attention rather than as an error at the
restart. The trigger is rare — a payload corrupt enough to fail the hash but intact
enough to still parse; anything that breaks framing is dropped and the job simply
expires — which is presumably why it went unnoticed.

The fix is a **replay**, not a rebuild. It rests on one observation: a job is strictly
sequential, so at most one pass is ever in flight, and when a hash fails every node is
in exactly one of two states — it computed that pass, or it did not. Nothing is more
than one pass out of step. What shipped:

- `NetworkJob.pass_idx`, a sequence number the origin increments for every pass it
  dispatches. Appended after `progress` (§5.5), so it is the first of the new fields.
  Numbering starts at **1**, which makes `0` an unambiguous "this peer predates the
  field" with no extra state to track.
- A `PassSequence` object per job per node (`src/language_pipes/jobs/pass_sequence.py`),
  reached as `Job.passes`. It holds the pass number, the highest number seen, the retry
  count, the `replaying` flag, the refusal reason, and the payloads the node forwarded.
  It decides; `Job.replay(saved)` applies the decision to `data` / `compute_step` /
  `current_layer`. Keeping the two apart is what stops `Job` from accreting the
  bookkeeping, and it is where `attempt` belongs in phase 2. Phase 1 applied the same
  pattern to the cache: `JobCache` (`src/language_pipes/jobs/job_cache.py`, `Job.caching`)
  holds the nine prompt-cache fields and the decisions that need only them, so `Job`
  gained one field instead of nine.
- The saved payload is the whole `JobData`, not just the hidden state: Gemma 4's
  `shared_kv_states` are mutated as the pass flows and the next node needs the
  post-mutation copy. It is keyed by the pass's **entry point**
  `(compute_step, current_layer)`, because a node can host two layer ranges of one pipe
  and be handed the same pass twice. Per entry point this is one chunk of hidden state —
  tens of KB for a decode token, a few hundred KB to ~1 MB for a 32-token prefill chunk
  — bounded by `max_node_jobs`. It is written in `_state_send`, which is exactly what
  went on the wire (an origin with no local layers never enters `_state_process_layers`
  and would otherwise have nothing to replay).
- On receive, a node compares. Same number under the same entry point: replay — the
  saved payload goes back out, the FSM short-circuits to `SEND`, the cache is not
  touched. `last + 1`, or the same number at a different entry point, or a first packet
  on a node that just joined the job: compute normally. Anything else: this node's cache
  cannot be right for the incoming pass; refuse, and the receiver cancels the job with
  the reason rather than computing garbage. `0` skips the check.
- The origin keeps its own payload too, saved under `(EMBED, 0)`. On a bounce
  (`data is None`, `compute_step == EMBED`, which no other packet to the origin has) it
  does not re-enter `_state_embed` at all: it resends that payload under the same
  `pass_idx`. That is what fixes the prefill defect — the unconditional `advance()` is
  never reached — and the decode defect follows from the replay rule on the upstream
  nodes. A bounce naming an older `pass_idx` is a second node reporting the same dead
  pass and is dropped; an unnumbered one is honored, since only one pass is ever live.
- A per-pass retry cap (three) then cancel with a reason, so a node whose serialization
  is deterministically broken cannot loop until the stale timer.

The decode case is worth tracing once: nodes before the corruption point already
appended the token's KV and now replay, so they do not append again; nodes after it
never saw the token and compute it once. Every cache ends up holding the position
exactly once, and the origin's `current_token` and the client's streamed text are
untouched. `job_processor/test_restart.py` asserts exactly this over two real `Job`s and
the real wire format, for a chunk lost outbound, a chunk lost on the return, the chunk
after a restart, and a decode token.

`attempt` is *not* part of this phase: a bounce is a replay against unchanged caches,
whereas a cache `MISS` (§4.4) is a genuine rebuild from token 0 against fresh ones, and
`attempt` exists for the latter. It lands in phase 2, on `PassSequence`.

Still outstanding from this phase: the hand-driven two-node run against real weights
with a deliberately corrupted packet. Do it before phase 2 starts leaning on `pass_idx`
for drift detection.

**Phase 1 — single-node correctness.** `PromptCache`, the keyed chain and its
per-process secret, the unauthenticated-server rule (§2.3), `ChunkState`
offset, `past_seen_tokens`, the per-job `JobCache` (`Job.caching`), the token
budget with admission and LRU eviction,
adopt/store on the origin only (end-model layers), both config fields + TUI + docs.
Reuse only when the origin hosts the whole pipe locally. Ships a real feature and
proves the resume-from-offset and budget paths without any protocol change.

**Phase 2 — distributed reuse.** `NetworkJob` tags including `attempt` and
`cache_reserve_tokens`; per-node admission; store-on-tag in layer nodes; `CacheStatus`
in all three flavors; rebuild-with-reuse-disabled on `MISS`, which bumps `attempt`,
resets `pass_idx` to 0 and gives every node a fresh cache. This phase needs the most
integration testing.

**Phase 3 — full OpenAI surface.** `prompt_cache_options`, explicit breakpoints,
breakpoint→offset mapping, `cache_write_tokens`, chat-completion
`stream_options.include_usage`.

**Phase 4 — optimizations.** Per-block snapshots for partial system-prompt reuse;
CPU demotion of cold entries; adopt-by-move under budget pressure (§4.7); derive a
default `max_cache_tokens` from the node memory limit and the hosted layer count
instead of a fixed number. (Publishing held-ID digests in DSN state was previously
listed here and is deliberately not being done — see §4.4.)

---

## 12. Decisions worth a second opinion

1. **Scope isolation vs. hit rate.** Keying over the API key and origin node ID
   means a fleet of clients sharing one large system prompt gets no shared cache
   unless they share an API key. That is the right default for this project's
   privacy posture, but if a deployment wants it, an opt-in
   `shared_prompt_cache = true` on the jobs server could drop the API key from
   the scope. Not in this plan — and note the keyed chain does not make it free:
   HMAC stops a *layer node* from confirming prompts, but dropping the API key
   re-opens the cross-tenant `cached_tokens` oracle at the *API*, which is a
   different adversary and unaffected by how the IDs are computed.
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
6. **The `restart_token` defects were confirmed, not hypothetical** (phase 0, now
   fixed). They were pre-existing and rare, so they did not block starting — but they
   went in the first commit rather than a footnote, because caching changes their blast
   radius from one bad request to a poisoned entry that outlives it. The fix is a replay
   of the last pass from per-node saved output rather than a rebuild: it costs no
   recompute, keeps every cache untouched, and needs one sequence number on the wire. A
   rebuild via `attempt` was the earlier draft; it is still the right tool for a cache
   `MISS`, which is why `attempt` remains in phase 2. One thing the plan did not
   anticipate: a node can host two layer ranges of one pipe, so "the last pass" has to
   be recorded per entry point, not once per job.
7. **`prompt_cache_key` carries isolation on unauthenticated nodes** (§2.3). This is
   a deliberate departure from OpenAI, where the field is a routing hint and
   isolation comes from the org boundary; here, with no API keys, it is the only
   thing separating callers. The plan does not enforce a minimum length or reject
   weak values — a request without a key is simply uncached, and a request with one
   is trusted to have chosen it well. The alternative considered was disabling
   caching outright whenever `api_keys` is empty, which is stricter but denies the
   feature to every single-user and trusted-LAN deployment, which is most of them.
   Worth revisiting if operators turn out to send `prompt_cache_key: "default"`.
