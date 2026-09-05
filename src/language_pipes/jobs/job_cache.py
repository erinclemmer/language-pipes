"""Per-job prompt-cache bookkeeping.

Everything one job knows about the prompt cache: what the request asked for,
which chain IDs its prompt hashes to, how much of the prefix it adopted, and
where its slice is due to be snapshotted. It holds no tensors and talks to no
store - `PromptCache` owns the entries and `JobProcessor` decides when to read
and write them - so this is only the state those decisions run on, kept in one
place instead of spread across nine fields of `Job`.

The KV state itself stays on `Job.cache`: it is the working cache for the pass
in flight, not cache bookkeeping, and the layer code writes into it directly.
"""

from typing import List, Optional, Tuple

from language_pipes.jobs.prompt_cache import BLOCK_SIZE, MIN_CACHE_TOKENS
from language_pipes.util.oai_cache import CacheOptions


class JobCache:
    # What the request asked for.
    options: CacheOptions
    # Binds this job's chain to one origin node, API key and `prompt_cache_key`.
    scope: bytes
    # Chain IDs by block index; `ids[i]` names the `i * BLOCK_SIZE` prefix.
    ids: List[bytes]
    # Prompt tokens whose keys and values arrived with an adopted entry rather
    # than being embedded by this job.
    prefix_len: int
    # Reported to the client as usage.*_tokens_details.cached_tokens.
    cached_tokens: int
    # Token counts at which this job's slice should be snapshotted.
    write_points: List[int]
    # The write point the chunk currently being embedded ends on, if any.
    pending_write_id: Optional[bytes]
    pending_write_tokens: int
    # Whether this job holds a reservation against the store's budget.
    reserved: bool

    def __init__(self, options: Optional[CacheOptions] = None):
        self.options = options if options is not None else CacheOptions()
        self.scope = b''
        self.ids = []
        self.prefix_len = 0
        self.cached_tokens = 0
        self.write_points = []
        self.pending_write_id = None
        self.pending_write_tokens = 0
        self.reserved = False

    def searched(self) -> bool:
        """Whether this job ever reached the store.

        `ids` is filled in only once the job is past every admission check, so
        it is what separates "looked and found nothing" from "never looked":
        caching off on the node, opted out by the request, a prompt too short to
        be worth it, a pipe with a remote segment, or a refused reservation.
        """
        return len(self.ids) > 0

    def forget(self):
        """Run the rest of the job uncached, reading nothing and writing nothing."""
        self.ids = []
        self.write_points = []
        self.clear_pending()

    def adopt(self, blocks: int):
        """Record a prefix taken from a stored entry, in blocks."""
        self.prefix_len = blocks * BLOCK_SIZE
        self.cached_tokens = self.prefix_len

    def plan_prompt_write(self, prompt_tokens: int):
        """Snapshot at the last block boundary of the prompt, in implicit mode.

        A boundary at or below what was just adopted is already stored, so there
        is nothing to add.
        """
        point = (prompt_tokens // BLOCK_SIZE) * BLOCK_SIZE
        if point >= MIN_CACHE_TOKENS and point > self.prefix_len:
            self.write_points = [point]

    def next_write_point(self, chunk_end: int) -> Optional[int]:
        """The write point a chunk ending at `chunk_end` completes, if any.

        Write points are block boundaries, and `BLOCK_SIZE` is a multiple of
        `CHUNK_SIZE`, so a chunk either lands exactly on one or on none.
        """
        for point in self.write_points:
            if point == chunk_end:
                return point
        return None

    def tag(self, chunk_end: int):
        """Mark the chunk about to be embedded if it ends on a write point.

        Cleared first: the origin is the only writer in Phase 1, so a tag that is
        not renewed before an embed must not survive into the next pass.
        """
        self.clear_pending()
        point = self.next_write_point(chunk_end)
        if point is None:
            return
        blocks = point // BLOCK_SIZE
        if blocks >= len(self.ids):
            return
        self.pending_write_id = self.ids[blocks]
        self.pending_write_tokens = point

    def take_pending(self) -> Optional[Tuple[bytes, int]]:
        """Claim the tag, if there is one. It is consumed either way: a write
        that turns out to be unsafe is dropped, not retried."""
        if self.pending_write_id is None:
            return None
        pending = (self.pending_write_id, self.pending_write_tokens)
        self.clear_pending()
        return pending

    def clear_pending(self):
        self.pending_write_id = None
        self.pending_write_tokens = 0

    def response_write_point(self, covered: int) -> Optional[int]:
        """The boundary to store the prompt-plus-answer entry at, if any.

        `covered` is how many positions the cache holds, which is one short of
        the sequence: the last sampled token was never embedded.
        """
        point = (covered // BLOCK_SIZE) * BLOCK_SIZE
        if point < MIN_CACHE_TOKENS or point <= max(self.write_points, default=0):
            return None
        return point

    def log_fields(self) -> str:
        """Cache outcome for the per-job completion line.

        Neither the API key nor `prompt_cache_key` is ever logged: on an
        unauthenticated node the cache key is what separates one caller from
        another, so writing it to a log file would undo that.
        """
        if not self.searched():
            return "cache=off cached=0"
        outcome = "hit" if self.cached_tokens > 0 else "miss"
        scope = self.scope[:4].hex() if len(self.scope) > 0 else ""
        return f"cache={outcome} cached={self.cached_tokens} scope={scope}"
