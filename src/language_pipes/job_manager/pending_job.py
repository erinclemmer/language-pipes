from promise import Promise
from typing import Callable

from transformers.cache_utils import DynamicCache

from language_pipes.job_manager.job import Job
from language_pipes.job_manager.chunk_state import ChunkState

class PendingJob:
    job: Job
    last_update: int
    resolve: Promise
    update: Callable[[Job], None]
    cache: DynamicCache
    chunking: ChunkState

    def __init__(
        self, 
        job: Job, 
        last_update: int, 
        resolve: Promise, 
        update: Callable[[Job], None],
        prompt_length: int = 0,
        chunk_size: int = 0
    ):
        self.job = job
        self.last_update = last_update
        self.resolve = resolve
        self.update = update
        self.cache = DynamicCache()
        self.chunking = ChunkState()
        if prompt_length > 0 and chunk_size > 0:
            self.chunking.init(prompt_length, chunk_size)
