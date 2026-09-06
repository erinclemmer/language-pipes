import logging
from time import time
from typing import Dict, List, Optional, Tuple
from time import sleep
from threading import Thread

from transformers import PretrainedConfig

from language_pipes.jobs.cache_policy import CacheOutcome, CachePolicy
from language_pipes.jobs.job import Job
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import PromptCache
from language_pipes.util.enums import JobStatus
from language_pipes.util.utils import release_memory

CHECK_JOB_INTERVAL = 10
EXPIRED_JOB_TIME = 60  # Unified timeout for both prefill and decode phases

class JobTracker:
    jobs_completed: List[str]
    jobs_pending: Dict[str, List[Job]]
    shutdown: bool
    # None when the node was built without a cache (tests, and any path that
    # does not go through ContentProvider.set_router).
    prompt_cache: Optional[PromptCache]

    def __init__(self, prompt_cache: Optional[PromptCache] = None):
        self.jobs_completed = []
        self.jobs_pending = { }
        self.shutdown = False
        self.prompt_cache = prompt_cache
        self.logger = logging.getLogger(__name__)
        Thread(target=self.check_stale_jobs, args=( )).start()

    def check_stale_jobs(self):
        while True:
            if self.shutdown:
                return
            for key in self.jobs_pending:
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
                    # Routed through `remove_job` so a stale job's entries are
                    # queued for demotion the same as any other ending job.
                    self.remove_job(job_id)

                if len(remove_jobs) > 0:
                    release_memory()

            if self.prompt_cache is not None:
                # A layer node is never told a job finished, so this is also
                # what demotes its entries - `sweep` drops anything none of
                # our still-pending jobs touched.
                live_ids: set = set()
                for job in self.get_jobs():
                    live_ids.update(job.caching.touched_ids)
                self.prompt_cache.sweep(live_ids)

            sleep(CHECK_JOB_INTERVAL)

    def get_job(self, job_id: str) -> Optional[Job]:
        for key in self.jobs_pending:
            for j in self.jobs_pending[key]:
                if j.job_id == job_id:
                    return j
        return None

    def get_jobs(self) -> List[Job]:
        jobs = []
        for key in list(self.jobs_pending.keys()):
            jobs.extend(self.jobs_pending[key])
        return jobs

    def jobs_for_pipes(self, pipe_ids: List[str]) -> List[Job]:
        return [j for j in self.get_jobs() if j.pipe_id in pipe_ids]

    def jobs_for_model(self, model_id: str, origin_node_id: Optional[str] = None) -> List[Job]:
        return [
            j for j in self.get_jobs()
            if j.model_id == model_id and (origin_node_id is None or j.origin_node_id == origin_node_id)
        ]

    def _release_reservation(self, job_id: str):
        """Give a finished job's share of the cache budget back."""
        if self.prompt_cache is not None:
            self.prompt_cache.release(job_id)

    def remove_job(self, job_id: str):
        removed: Optional[Job] = None
        for key in list(self.jobs_pending.keys()):
            remaining = []
            for j in self.jobs_pending[key]:
                if j.job_id == job_id:
                    removed = j
                else:
                    remaining.append(j)
            self.jobs_pending[key] = remaining
        self._release_reservation(job_id)
        # Nothing still running needs this job's entries on the device, so
        # queue them for demotion - the actual move happens off this thread.
        if removed is not None and self.prompt_cache is not None:
            self.prompt_cache.demote_for_job(removed)

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

    def add_job(
        self,
        network_job: NetworkJob,
        config: PretrainedConfig,
        cache_policy: CachePolicy,
        model_id: str = "",
    ) -> Tuple[Optional[Job], CacheOutcome]:
        existing = self.get_job(network_job.job_id)
        assert existing is None
        
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
            return None, CacheOutcome.OK

        if network_job.data.state is None:
            raise Exception("job should be embedded before adding a pending job")

        # After the checks above, so a refused packet cannot leave a reservation
        # behind for a job that never ran.
        outcome = cache_policy.adopt_for_node(job, network_job)
        if outcome == CacheOutcome.MISS:
            return None, outcome

        if 'network' not in self.jobs_pending:
            self.jobs_pending['network'] = []
        
        self.jobs_pending['network'].append(job)
        return job, outcome
