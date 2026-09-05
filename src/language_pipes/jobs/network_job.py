from language_pipes.util.byte_helper import ByteHelper
from language_pipes.util.enums import ComputeStep
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_time import JobTime
from language_pipes.jobs.completed_pass import CompletedPass
from language_pipes.jobs.job_progress import JobProgress

class NetworkJob:
    job_id: str
    pipe_id: str
    origin_node_id: str
    current_layer: int
    compute_step: ComputeStep
    data: JobData | None
    data_hash: bytes
    times: list[JobTime]
    completed: CompletedPass | None
    progress: JobProgress | None
    pass_idx: int
    # Bumped by the origin every time it restarts this job from token 0, so a
    # packet from a dead attempt can be told apart from the retry that replaced
    # it. See `documentation/architecture.md`, "Prompt cache across nodes".
    attempt: int
    # Prompt-cache tags. `cache_use_*` rides the first pass and tells each node
    # which stored prefix to adopt; `cache_write_*` names the boundary the pass
    # in flight should be snapshotted at; `cache_reserve_tokens` is the origin's
    # estimate of what the job will ask each node's cache to hold.
    cache_use_id: bytes
    cache_use_tokens: int
    cache_write_id: bytes
    cache_write_tokens: int
    cache_reserve_tokens: int
    prefill_chunk_size: int

    def __init__(
        self,
        job_id: str,
        pipe_id: str,
        origin_node_id: str,
        current_layer: int,
        data: JobData | None,
        data_hash: bytes,
        compute_step: ComputeStep,
        times: list[JobTime],
        completed: CompletedPass | None = None,
        progress: JobProgress | None = None,
        pass_idx: int = 0,
        attempt: int = 0,
        cache_use_id: bytes = b'',
        cache_use_tokens: int = 0,
        cache_write_id: bytes = b'',
        cache_write_tokens: int = 0,
        cache_reserve_tokens: int = 0
    ):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.origin_node_id = origin_node_id
        self.current_layer = current_layer
        self.data = data
        self.data_hash = data_hash
        self.compute_step = compute_step
        self.times = times
        self.completed = completed
        self.progress = progress
        self.pass_idx = pass_idx
        self.attempt = attempt
        self.cache_use_id = cache_use_id
        self.cache_use_tokens = cache_use_tokens
        self.cache_write_id = cache_write_id
        self.cache_write_tokens = cache_write_tokens
        self.cache_reserve_tokens = cache_reserve_tokens

    def to_bytes(self):
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.origin_node_id)
        bts.write_int(self.current_layer)
        bts.write_int(self.compute_step.value)
        bts.write_bytes(self.data.to_bytes() if self.data is not None else b'')
        bts.write_bytes(self.data_hash)

        bts.write_int(len(self.times))
        for time in self.times:
            bts.write_bytes(time.to_bytes())

        bts.write_bytes(self.completed.to_bytes() if self.completed is not None else b'')
        bts.write_bytes(self.progress.to_bytes() if self.progress is not None else b'')
        # Appended fields keep their order: a peer that predates one reads the
        # zero/empty value that `ByteHelper` gives at EOF.
        bts.write_int(self.pass_idx)
        bts.write_int(self.attempt)
        bts.write_bytes(self.cache_use_id)
        bts.write_int(self.cache_use_tokens)
        bts.write_bytes(self.cache_write_id)
        bts.write_int(self.cache_write_tokens)
        bts.write_int(self.cache_reserve_tokens)

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)

        job_id = bts.read_string()
        pipe_id = bts.read_string()
        origin_node_id = bts.read_string()
        current_layer = bts.read_int()
        step = ComputeStep(bts.read_int())
        job_bytes = bts.read_bytes()
        job_data = JobData.from_bytes(job_bytes) if job_bytes != b'' else None
        data_hash = bts.read_bytes()

        valid = True
        if data_hash != b'':
            valid = JobData.validate_state(job_bytes, data_hash)

        times = []
        for _ in range(0, bts.read_int()):
            times.append(JobTime.from_bytes(bts.read_bytes()))

        # Both absent when the peer runs a build that does not report them
        completed_bytes = bts.read_bytes()
        completed = CompletedPass.from_bytes(completed_bytes) if completed_bytes != b'' else None

        progress_bytes = bts.read_bytes()
        progress = JobProgress.from_bytes(progress_bytes) if progress_bytes != b'' else None

        # 0 means the peer does not number passes; the origin numbers from 1.
        pass_idx = bts.read_int()

        # A peer that predates the cache protocol reads attempt 0 and empty
        # tags, which is exactly "never restarted, nothing to adopt or store".
        attempt = bts.read_int()
        cache_use_id = bts.read_bytes()
        cache_use_tokens = bts.read_int()
        cache_write_id = bts.read_bytes()
        cache_write_tokens = bts.read_int()
        cache_reserve_tokens = bts.read_int()

        return NetworkJob(
            job_id=job_id,
            pipe_id=pipe_id,
            origin_node_id=origin_node_id,
            current_layer=current_layer,
            data=job_data,
            data_hash=data_hash,
            compute_step=step,
            times=times,
            completed=completed,
            progress=progress,
            pass_idx=pass_idx,
            attempt=attempt,
            cache_use_id=cache_use_id,
            cache_use_tokens=cache_use_tokens,
            cache_write_id=cache_write_id,
            cache_write_tokens=cache_write_tokens,
            cache_reserve_tokens=cache_reserve_tokens
        ), valid
