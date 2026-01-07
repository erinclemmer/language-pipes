import gc
import ctypes
import torch
from typing import List, Optional
from time import time, sleep
from threading import Thread

from language_pipes.jobs.job import Job
from language_pipes.jobs.layer_job import LayerJob
from language_pipes.jobs.pending_job import PendingJob

CHECK_JOB_INTERVAL = 10
EXPIRED_JOB_TIME = 60  # Unified timeout for both prefill and decode phases

try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except:
    _malloc_trim = None

class JobTracker:
    jobs_completed: List[str]
    jobs_pending: List[PendingJob]

    def __init__(self, logger):
        self.logger = logger
        self.jobs_completed = []
        self.jobs_pending = []
        Thread(target=self.check_stale_jobs, args=( )).start()

    def check_stale_jobs(self):
        while True:
            remove_jobs = []
            for j in self.jobs_pending:
                stale_time = time() - j.last_update
                # Unified timeout - prefill chunks regularly update last_update,
                # so both prefill and decode phases use the same timeout
                if stale_time > EXPIRED_JOB_TIME:
                    self.logger.warning(
                        f"[Stale] job={j.job.job_id[:8]} timed out after {stale_time:.1f}s "
                        f"(token={j.job.current_token})"
                    )
                    remove_jobs.append(j.job.job_id)

            if len(remove_jobs) == 0:
                sleep(CHECK_JOB_INTERVAL)
                continue
        
            for job_id in remove_jobs:
                self.jobs_pending = [j for j in self.jobs_pending if j.job.job_id != job_id]
            
            gc.collect()
            torch.cuda.empty_cache()
            if _malloc_trim is not None:
                _malloc_trim(0)

            sleep(CHECK_JOB_INTERVAL)

    def get_pending_job(self, job_id: str) -> Optional[PendingJob]:
        for j in self.jobs_pending:
            if j.job.job_id == job_id:
                return j
        return None

    def send_job_update(self, job: Job):
        job_id = job.job_id
        if job_id in self.jobs_completed:
            return
        pending_job = self.get_pending_job(job_id)
        if pending_job is None or pending_job.update is None:
            return
        self.logger.info(f'Received job update for {job_id}\n')
        pending_job.last_update = time()
        return pending_job.update(job)

    def complete_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.jobs_completed:
            return
        self.jobs_completed.append(job_id)
        pending_job = self.get_pending_job(job_id)
        if pending_job is None or pending_job.resolve is None:
            return
        self.logger.info(f'Received job complete for {job_id}\n')
        pending_job.resolve(job)
        self.jobs_pending = [j for j in self.jobs_pending if j.job.job_id != job_id]

    def update_job_time(self, job_id: str):
        """Update the last_update time for a pending job to prevent stale timeout."""
        pending_job = self.get_pending_job(job_id)
        if pending_job is None:
            return
        pending_job.last_update = time()

    def add_pending_job(self, layer_job: LayerJob):
        existing = self.get_pending_job(layer_job.job_id)
        if existing is not None:
            return existing  # Return existing job instead of None
        job = Job(
            "",
            layer_job.origin_node_id, 
            0, [], layer_job.pipe_id, ""
        )
        job.job_id = layer_job.job_id
        if layer_job.data.state is None:
            raise Exception("job should be embedded before adding a pending job")
        job.prompt_tokens = layer_job.data.state.size()[1]
        pending_job = PendingJob(job, time(), None, None)
        self.jobs_pending.append(pending_job)
        return pending_job