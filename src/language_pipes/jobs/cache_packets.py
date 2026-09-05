"""The prompt cache's back-channel.

Everything the cache needs to say *forward* rides the job packet itself: which
prefix to adopt, which boundary to snapshot. What has nowhere to ride is the
negatives. A job packet only ever goes on to the node hosting the next layers,
or back to the origin at `HEAD`, so a node that cannot compute has nothing to
forward, and a node that computed fine but cannot store has no field in the
outgoing packet to say so.

One packet type covers all three, sent under `CACHE_PROTOCOL` and modeled on
`jobs/job_cancel.py`. Each carries the `attempt` it refers to, because the
answer to a dead attempt must not act on the retry that replaced it.
"""

from enum import IntEnum

from language_pipes.util.byte_helper import ByteHelper


class CacheReason(IntEnum):
    # node -> origin: "I do not hold the prefix you told me to adopt, and I did
    # not compute this pass." The origin rebuilds the job with reuse off.
    MISS = 0
    # node -> origin: "I computed fine, but I have no cache budget for this
    # job." The origin stops tagging write points, so no node stores a prefix
    # this one would be missing.
    NO_STORE = 1
    # origin -> nodes: "Drop this job; that attempt is dead." An optimization,
    # not a correctness requirement - `attempt` on the job packet is what makes
    # the rebuild safe - but it frees the stale caches now instead of at the
    # next packet.
    ABORT = 2


class CacheStatus:
    job_id: str
    pipe_id: str
    attempt: int
    reason: CacheReason

    def __init__(self, job_id: str, pipe_id: str, attempt: int, reason: CacheReason):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.attempt = attempt
        self.reason = reason

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_int(self.attempt)
        bts.write_int(int(self.reason))
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> 'CacheStatus':
        bts = ByteHelper(data)
        return CacheStatus(
            job_id=bts.read_string(),
            pipe_id=bts.read_string(),
            attempt=bts.read_int(),
            reason=CacheReason(bts.read_int())
        )
