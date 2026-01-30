import gc
import ctypes
import torch
from time import time
from typing import List, Optional
from time import time, sleep
from threading import Thread

from language_pipes.jobs.job import Job
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.config import LpConfig

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
    jobs_pending: List[Job]

    def __init__(self, logger, config: LpConfig):
        self.logger = logger
        self.config = config
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
                        f"[Stale] job={j.job_id[:8]} timed out after {stale_time:.1f}s "
                        f"(token={j.current_token})"
                    )
                    remove_jobs.append(j.job_id)

            if len(remove_jobs) == 0:
                sleep(CHECK_JOB_INTERVAL)
                continue
        
            for job_id in remove_jobs:
                self.jobs_pending = [j for j in self.jobs_pending if j.job_id != job_id]
            
            gc.collect()
            torch.cuda.empty_cache()
            if _malloc_trim is not None:
                _malloc_trim(0)

            sleep(CHECK_JOB_INTERVAL)

    def get_job(self, job_id: str) -> Optional[Job]:
        for j in self.jobs_pending:
            if j.job_id == job_id:
                return j
        return None

    def complete_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.jobs_completed:
            return
        self.jobs_completed.append(job_id)
        if job.resolve is None:
            return
        self.logger.info(f'Received job complete for {job_id}\n')
        self.logger.info("[TIMING] Prefill Summary:")
        job.timing_stats.prefill_times.log_summary(self.logger)
        self.logger.info("[TIMING] Output Summary:")
        job.timing_stats.output_times.log_summary(self.logger)
        job.resolve(job)
        self.jobs_pending = [j for j in self.jobs_pending if j.job_id != job_id]

    def update_job_time(self, job_id: str):
        """Update the last_update time for a pending job to prevent stale timeout."""
        job = self.get_job(job_id)
        if job is None:
            return
        job.last_update = time()

    def add_job(self, network_job: NetworkJob) -> Job | None:
        existing = self.get_job(network_job.job_id)
        if existing is not None:
            return None
        
        job = Job(
            origin_node_id=network_job.origin_node_id,
            messages=[],
            model_id="",
            prefill_chunk_size=self.config.prefill_chunk_size,
            pipe_id=network_job.pipe_id,
            data=network_job.data
        )
        job.job_id = network_job.job_id
        
        if network_job.data.state is None:
            raise Exception("job should be embedded before adding a pending job")
        
        job.prompt_tokens = network_job.data.state.size()[1]
        job.last_update = time()
        self.jobs_pending.append(job)
        return job
