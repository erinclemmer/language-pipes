from language_pipes.util.byte_helper import ByteHelper
from language_pipes.jobs.job_time import JobTime

class CompletedPass:
    """A finished forward pass - one decode token or one prefill chunk - with the
    timings every node it visited contributed.

    Only the origin node runs the head, so only it can tell where a pass ends.
    It ships the pass it just closed out alongside the next one so nodes without
    the end model can report the same prefill/decode speeds instead of nothing.
    """

    index: int
    token_count: int
    is_prefill: bool
    times: list[JobTime]

    def __init__(self, index: int, token_count: int, is_prefill: bool, times: list[JobTime]):
        self.index = index
        self.token_count = token_count
        self.is_prefill = is_prefill
        self.times = times

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_int(self.index)
        bts.write_int(self.token_count)
        bts.write_int(1 if self.is_prefill else 0)
        bts.write_int(len(self.times))
        for time in self.times:
            bts.write_bytes(time.to_bytes())
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> "CompletedPass":
        bts = ByteHelper(data)
        index = bts.read_int()
        token_count = bts.read_int()
        is_prefill = bts.read_int() == 1
        times = []
        for _ in range(0, bts.read_int()):
            times.append(JobTime.from_bytes(bts.read_bytes()))
        return CompletedPass(
            index=index,
            token_count=token_count,
            is_prefill=is_prefill,
            times=times
        )
