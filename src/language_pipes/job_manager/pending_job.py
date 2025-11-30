from promise import Promise
from typing import Callable

from language_pipes.job_manager.job import Job

class PendingJob:
    job: Job
    resolve: Promise
    update: Callable[[Job], None]

    def __init__(self, job: str, resolve: Promise, update: Callable[[Job], None]):
        self.job = job
        self.resolve = resolve
        self.update = update