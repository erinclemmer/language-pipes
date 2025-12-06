from promise import Promise
from typing import Callable

from language_pipes.job_manager.job import Job

class PendingJob:
    job: Job
    last_update: int
    resolve: Promise
    update: Callable[[Job], None]

    def __init__(self, job: str, last_update: int, resolve: Promise, update: Callable[[Job], None]):
        self.job = job
        self.last_update = last_update
        self.resolve = resolve
        self.update = update