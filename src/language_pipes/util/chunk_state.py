from time import time

class ChunkState:
    """Tracks prefill chunking state for a job (local only, not sent over network)."""
    current_chunk: int  # Current chunk index being processed (0-based)
    total_chunks: int  # Total chunks for prefill (0 = no chunking needed)
    chunk_size: int  # Size of each chunk
    prompt_length: int  # Total prompt length

    def __init__(self):
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0
        self.prompt_length = 0

    def init(self, prompt_length: int, chunk_size: int):
        """Initialize chunking if the prompt exceeds chunk_size."""
        self.prompt_length = prompt_length
        if prompt_length > chunk_size:
            self.chunk_size = chunk_size
            self.total_chunks = (prompt_length + chunk_size - 1) // chunk_size
            self.current_chunk = 0
        else:
            self.total_chunks = 0
            self.current_chunk = 0
            self.chunk_size = 0

    def is_active(self) -> bool:
        """Returns True if prefill chunking is active."""
        return self.total_chunks > 1

    def has_more(self) -> bool:
        """Returns True if there are more chunks to process."""
        return self.is_active() and self.current_chunk < self.total_chunks - 1

    def is_final(self) -> bool:
        """Returns True if currently processing the final chunk."""
        return not self.is_active() or self.current_chunk == self.total_chunks - 1

    def get_range(self) -> tuple[int, int]:
        """Get the (start, end) token indices for the current chunk."""
        if not self.is_active():
            return (0, self.prompt_length)
        start = self.current_chunk * self.chunk_size
        end = min(start + self.chunk_size, self.prompt_length)
        return (start, end)

    def advance(self):
        """Move to the next chunk."""
        self.current_chunk += 1

    def print_start(self, logger):
        if self.is_active():
            logger.info(
                f"prompt_tokens={self.prompt_length}, "
                f"chunks={self.total_chunks}, "
                f"chunk_size={self.chunk_size}"
            )
        else:
            logger.info(f"prompt_tokens={self.prompt_length} (no chunking)")

    def disable(self):
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0

    def __str__(self) -> str:
        """String representation for logging."""
        if not self.is_active():
            return f"ChunkState(inactive, prompt_length={self.prompt_length})"
        return (
            f"ChunkState(chunk={self.current_chunk + 1}/{self.total_chunks}, "
            f"chunk_size={self.chunk_size}, prompt_length={self.prompt_length})"
        )

def log_prefill_chunk_complete(logger, job: "Job") -> None:
    chunk_time_ms = (time() - job.chunk_start_time) * 1000
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {job.chunking.current_chunk + 1}/"
        f"{job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
    )

def log_prefill_chunk_start(logger, job: "Job", chunk_start: int, chunk_end: int) -> None:
    job.chunk_start_time = time()
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {job.chunking.current_chunk + 1}/"
        f"{job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
    )

def log_prefill_summary(logger, job: "Job") -> None:
    total_prefill_ms = (time() - job.prefill_start_time) * 1000
    tokens_per_sec = (job.prompt_tokens / total_prefill_ms) * 1000 if total_prefill_ms > 0 else 0
    logger.info(
        f"[Prefill] job={job.job_id[:8]} finished: "
        f"prompt_tokens={job.prompt_tokens}, "
        f"total_time={total_prefill_ms:.1f}ms, "
        f"throughput={tokens_per_sec:.1f} tok/s"
    )