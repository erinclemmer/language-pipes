from time import time
from promise import Promise
from typing import Callable

from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job import Job
from language_pipes.util.chunk_state import ChunkState

class PendingJob:
    job: Job
    last_update: float
    cache: DynamicCache
    chunking: ChunkState
    resolve: Promise | None
    update: Callable[[Job], None] | None

    def __init__(
        self, 
        job: Job, 
        last_update: float, 
        resolve: Promise | None, 
        update: Callable[[Job], None] | None,
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

    def set_last_update(self):
        self.last_update = time()