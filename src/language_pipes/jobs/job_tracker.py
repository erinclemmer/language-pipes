import gc
import ctypes
import logging
import torch
from time import time
from typing import Dict, List, Optional
from time import sleep
from threading import Thread

from transformers import PretrainedConfig

from language_pipes.jobs.job import Job
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.util.enums import JobStatus

CHECK_JOB_INTERVAL = 10
EXPIRED_JOB_TIME = 60  # Unified timeout for both prefill and decode phases

try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except:  # noqa: E722
    _malloc_trim = None

class JobTracker:
    jobs_completed: List[str]
    jobs_pending: Dict[str, List[Job]]
    shutdown: bool

    def __init__(self):
        self.jobs_completed = []
        self.jobs_pending = { }
        self.shutdown = False
        self.logger = logging.getLogger(__name__)
        Thread(target=self.check_stale_jobs, args=( )).start()

    def check_stale_jobs(self):
        while True:
            if self.shutdown:
                return
            for key in self.jobs_pending.keys():
                remove_jobs = []
                for j in self.jobs_pending[key]:
                    if j.stale:
                        remove_jobs.append(j.job_id)
                        continue
                    stale_time = time() - j.last_update
                    # Unified timeout - prefill chunks regularly update last_update,
                    # so both prefill and decode phases use the same timeout
                    if stale_time > EXPIRED_JOB_TIME:
                        remove_jobs.append(j.job_id)

                for job_id in remove_jobs:
                    self.jobs_pending[key] = [j for j in self.jobs_pending[key] if j.job_id != job_id]

                if len(remove_jobs) > 0:        
                    gc.collect()
                    torch.cuda.empty_cache()
                    if _malloc_trim is not None:
                        _malloc_trim(0)

            sleep(CHECK_JOB_INTERVAL)

    def get_job(self, job_id: str) -> Optional[Job]:
        for key in self.jobs_pending:
            for j in self.jobs_pending[key]:
                if j.job_id == job_id:
                    return j
        return None

    def get_jobs(self) -> List[Job]:
        jobs = []
        for key in self.jobs_pending:
            jobs.extend(self.jobs_pending[key])
        return jobs

    def jobs_for_pipes(self, pipe_ids: List[str]) -> List[Job]:
        return [j for j in self.get_jobs() if j.pipe_id in pipe_ids]

    def jobs_for_model(self, model_id: str, origin_node_id: Optional[str] = None) -> List[Job]:
        return [
            j for j in self.get_jobs()
            if j.model_id == model_id and (origin_node_id is None or j.origin_node_id == origin_node_id)
        ]

    def remove_job(self, job_id: str):
        for key in list(self.jobs_pending.keys()):
            self.jobs_pending[key] = [j for j in self.jobs_pending[key] if j.job_id != job_id]

    def complete_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.jobs_completed:
            return

        self.jobs_completed.append(job_id)

        if job.resolve is not None:
            job.resolve(job) # pyright: ignore[reportCallIssue]

        self.remove_job(job_id)

    def cancel_job(self, job: Job, reason: str):
        """Stop a job now instead of leaving it to time out.

        Marks it so any in-flight processing halts at the next checkpoint, then
        completes it so an API caller waiting on the promise gets an error back
        rather than a hung request.
        """
        if job.job_id in self.jobs_completed:
            return

        job.stale = True
        job.cancel_reason = reason
        job.status = JobStatus.ERROR
        self.logger.info(f"Job {job.job_id[:4]} canceled: {reason}")
        self.complete_job(job)

    def update_job_time(self, job_id: str):
        """Update the last_update time for a pending job to prevent stale timeout."""
        job = self.get_job(job_id)
        if job is None:
            return
        job.last_update = time()

    def add_job(self, network_job: NetworkJob, config: PretrainedConfig, model_id: str = "") -> Job | None:
        existing = self.get_job(network_job.job_id)
        if existing is not None:
            return None

        job = Job(
            origin_node_id=network_job.origin_node_id,
            messages=[],
            model_id=model_id,
            pipe_id=network_job.pipe_id,
            data=network_job.data,
            config=config
        )
        job.job_id = network_job.job_id
        
        if network_job.data is None:
            return

        if network_job.data.state is None:
            raise Exception("job should be embedded before adding a pending job")

        # prompt_tokens is left at 0: only the origin tokenizes, and the state in
        # flight is one pass wide, not the prompt. The UI reads the origin's own
        # count out of Job.display_progress() instead.
        job.last_update = time()
        if 'network' not in self.jobs_pending:
            self.jobs_pending['network'] = []
        
        self.jobs_pending['network'].append(job)
        return job
