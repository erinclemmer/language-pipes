from enum import IntEnum

from language_pipes.util.byte_helper import ByteHelper

# IntEnum for easier wire transport
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
