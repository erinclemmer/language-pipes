import random
import threading
from time import sleep
from typing import Callable

from language_pipes.jobs.network_job import NetworkJob

class JobQueue:
    _lock: threading.Lock
    _is_shutdown: Callable[[], bool]
    _get_max_node_jobs: Callable[[], int]
    _queue: dict[str, list[NetworkJob]]

    def __init__(self, 
        is_shutdown: Callable[[], bool],
        get_max_node_jobs: Callable[[], int]
    ):
        self._lock = threading.Lock()
        self._queue = { }
        self._is_shutdown = is_shutdown
        self._get_max_node_jobs = get_max_node_jobs

    def wait_for_job(self) -> NetworkJob | None:
        """Wait for a job from the queue. Returns None if shutting down."""
        while True:
            if self._is_shutdown():
                return None
            if len(self._queue.keys()) > 0:
                with self._lock:
                    node_id = random.choice(list(self._queue.keys()))
                    node_jobs = self._queue[node_id]
                    idx = random.randrange(len(node_jobs))
                    network_job = self._queue[node_id].pop(idx)
                    if len(self._queue[node_id]) == 0:
                        del self._queue[node_id]
                return network_job
            sleep(0.01)

    def drop_queued(self, job_id: str):
        """Discard packets for a job that is no longer running."""
        with self._lock:
            for node_id in list(self._queue.keys()):
                self._queue[node_id] = [j for j in self._queue[node_id] if j.job_id != job_id]
                if len(self._queue[node_id]) == 0:
                    del self._queue[node_id]

    def _is_in_queue(self, node_id: str, job_id: str) -> bool:
        if node_id not in self._queue:
            return False
        
        return any(j.job_id == job_id for j in self._queue[node_id])

    def add_to_queue(self, node_id: str, job: NetworkJob):
        if self._is_in_queue(node_id, job.job_id):
            return

        with self._lock:
            if node_id not in self._queue:
                self._queue[node_id] = [ ]
            if len(self._queue[node_id]) > self._get_max_node_jobs():
                raise Exception("Maximum number of jobs for node reached")
            self._queue[node_id].insert(0, job)