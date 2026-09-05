from language_pipes.util.utils import CHUNK_SIZE

class ChunkState:
    job_id: str
    current_chunk: int  # Current chunk index being processed (0-based)
    total_chunks: int  # Total chunks for prefill (0 = no chunking needed)
    chunk_size: int  # Size of each chunk
    prompt_length: int  # Total prompt length
    start_offset: int  # Prompt tokens already in the cache before this job ran

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0
        self.prompt_length = 0
        self.start_offset = 0

    def init(self, prompt_length: int, start_offset: int = 0):
        """Initialize chunking for the part of the prompt this job must embed.

        `start_offset` is a prefix already covered by an adopted prompt-cache
        entry: its keys and values are in the cache, so the chunks start after
        it. Everything below is sized on the remaining suffix.
        """
        self.prompt_length = prompt_length
        self.start_offset = start_offset
        remaining = prompt_length - start_offset
        if remaining > CHUNK_SIZE:
            self.total_chunks = (remaining + CHUNK_SIZE - 1) // CHUNK_SIZE
            self.current_chunk = 0
            self.chunk_size = CHUNK_SIZE
        else:
            self.total_chunks = 0
            self.current_chunk = 0
            self.chunk_size = 0

    def is_active(self) -> bool:
        return self.total_chunks > 1

    def has_more(self) -> bool:
        return self.is_active() and self.current_chunk < self.total_chunks - 1

    def is_final(self) -> bool:
        return not self.is_active() or self.current_chunk == self.total_chunks - 1

    def get_range(self) -> tuple[int, int]:
        if not self.is_active():
            return (self.start_offset, self.prompt_length)
        start = self.start_offset + self.current_chunk * self.chunk_size
        end = min(start + self.chunk_size, self.prompt_length)
        return (start, end)

    def get_tokens_processed(self) -> int:
        """Prompt tokens covered by chunks of *this job* that have finished.

        The adopted prefix is not counted here - `Job.past_seen_tokens` adds it
        back, so this stays the answer to "how much of the work I was given have
        I done".
        """
        return self.get_range()[0] - self.start_offset

    def get_chunk_length(self) -> int:
        """Number of prompt tokens covered by the chunk currently being processed."""
        start, end = self.get_range()
        return end - start

    def advance(self):
        self.current_chunk += 1

    def disable(self):
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0
        self.start_offset = 0
