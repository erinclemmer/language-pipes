from language_pipes.util.utils import CHUNK_SIZE

class ChunkState:
    job_id: str
    current_chunk: int  # Current chunk index being processed (0-based)
    total_chunks: int  # Total chunks for prefill (0 = no chunking needed)
    chunk_size: int  # Size of each chunk
    prompt_length: int  # Total prompt length

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0
        self.prompt_length = 0

    def init(self, prompt_length: int):
        """Initialize chunking if the prompt exceeds chunk_size."""
        self.prompt_length = prompt_length
        if prompt_length > CHUNK_SIZE:
            self.total_chunks = (prompt_length + CHUNK_SIZE - 1) // CHUNK_SIZE
            self.current_chunk = 0
        else:
            self.total_chunks = 0
            self.current_chunk = 0
            self.chunk_size = 0

    def is_active(self) -> bool:
        return self.total_chunks > 1

    def has_more(self) -> bool:
        return self.is_active() and self.current_chunk < self.total_chunks

    def is_final(self) -> bool:
        return not self.is_active() or self.current_chunk == self.total_chunks - 1

    def get_range(self) -> tuple[int, int]:
        if not self.is_active():
            return (0, self.prompt_length)
        start = self.current_chunk * self.chunk_size
        end = min(start + self.chunk_size, self.prompt_length)
        return (start, end)

    def advance(self):
        self.current_chunk += 1

    def disable(self):
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_size = 0
