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

    def __str__(self) -> str:
        """String representation for logging."""
        if not self.is_active():
            return f"ChunkState(inactive, prompt_length={self.prompt_length})"
        return (
            f"ChunkState(chunk={self.current_chunk + 1}/{self.total_chunks}, "
            f"chunk_size={self.chunk_size}, prompt_length={self.prompt_length})"
        )
