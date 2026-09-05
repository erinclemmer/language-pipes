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

from language_pipes.jobs.network_job import NetworkJob
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
    # Set when a node on the pipe reports it has no budget for this job. An
    # entry the rest of the pipe holds and it does not is worse than no entry:
    # the next request adopts it, that node misses, and the job rebuilds.
    writes_stopped: bool
    # The tags the first pass carries to the other nodes on the pipe: which
    # entry each of them should adopt, and what to reserve against its budget.
    # Only the first pass carries them - after that every node has its job -
    # so the ID travels no further than it has to.
    use_id: bytes
    use_tokens: int
    reserve_tokens: int

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
        self.writes_stopped = False
        self.use_id = b''
        self.use_tokens = 0
        self.reserve_tokens = 0

    def searched(self) -> bool:
        """Whether this job ever reached the store.

        `ids` is filled in only once the job is past every admission check, so
        it is what separates "looked and found nothing" from "never looked":
        caching off on the node, opted out by the request, a prompt too short to
        be worth it, or a refused reservation.
        """
        return len(self.ids) > 0

    def forget(self):
        """Run the rest of the job uncached, reading nothing and writing nothing."""
        self.ids = []
        self.write_points = []
        self.use_id = b''
        self.use_tokens = 0
        self.reserve_tokens = 0
        self.clear_pending()

    def stop_writing(self):
        """Give up on storing anything for this job.

        Sent by a node that computed the pass fine but has no budget for it: an
        entry the rest of the pipe holds and it does not is memory spent on a
        prefix that is guaranteed to miss.
        """
        self.writes_stopped = True
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

        Cleared first: a tag not renewed before an embed must not survive into
        the next pass. On the origin this call is the only thing that sets one;
        on a layer node `tag_from_wire` is, and it clears in the same way.
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

    def pending(self) -> Optional[Tuple[bytes, int]]:
        """The boundary this pass is due to be snapshotted at, if any.

        Read, not claimed: the tag has to survive the store because the packet
        carrying it goes on to the next node, which stores its own slice under
        the same ID. It is cleared by the next `tag` or `tag_from_wire`, which
        is once per pass on every node.
        """
        if self.pending_write_id is None:
            return None
        return (self.pending_write_id, self.pending_write_tokens)

    def clear_pending(self):
        self.pending_write_id = None
        self.pending_write_tokens = 0

    def tag_from_wire(self, network_job: NetworkJob):
        """Take the write tag off an incoming packet.

        A layer node is told where the boundaries are - it never sees tokens -
        so the tag on the packet is the whole of its write path. Absent tag
        means no write, which is also how a tag is prevented from surviving
        into the next pass.
        """
        if network_job.cache_write_id == b'':
            self.clear_pending()
            return
        self.pending_write_id = network_job.cache_write_id
        self.pending_write_tokens = network_job.cache_write_tokens

    def use_from_wire(self, network_job: NetworkJob):
        """Carry the read tags onward, so the rest of the pipe sees them too.

        The origin only sends them on the first pass, and a node joins the job
        on the packet that carries them, so each node has to pass on what it was
        given for the nodes after it.
        """
        self.use_id = network_job.cache_use_id
        self.use_tokens = network_job.cache_use_tokens
        self.reserve_tokens = network_job.cache_reserve_tokens

    def add_write_point(self, point: int) -> bool:
        """Record another boundary to snapshot at, if it is worth recording.

        Used while decoding, where each pass pushes the sequence one token
        further and the answer crosses a boundary every `BLOCK_SIZE` tokens.
        Boundaries only ever grow, so a point at or below one already planned is
        already covered.
        """
        if self.writes_stopped:
            return False
        if point < MIN_CACHE_TOKENS or point <= max(self.write_points, default=0):
            return False
        self.write_points.append(point)
        return True

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
