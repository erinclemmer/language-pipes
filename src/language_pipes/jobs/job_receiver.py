import logging
from threading import Thread
from typing import Callable

from language_pipes.jobs.cache_protocol import CacheProtocol
from language_pipes.jobs.job_queue import JobQueue
from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.cache_packets import CacheReason, CacheStatus
from language_pipes.jobs.cache_policy import CacheOutcome, CachePolicy
from language_pipes.jobs.job import ComputeStep, Job
from language_pipes.jobs.job_cancel import CancelProtocol
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.jobs.job_processor import JobProcessor, JobContext


class JobReceiver:
    job_factory: JobFactory
    job_queue: JobQueue
    pipe_manager: PipeManager
    model_manager: ModelManager
    cancel_protocol: CancelProtocol
    shutdown: bool
    is_shutdown: Callable[[], bool]

    def __init__(
            self,
            job_factory: JobFactory,
            job_tracker: JobTracker,
            pipe_manager: PipeManager,
            model_manager: ModelManager,
            is_shutdown: Callable[[], bool],
            get_max_node_jobs: Callable[[], int]
    ):
        self.shutdown = False
        self.job_queue = JobQueue(lambda: (is_shutdown() or self.shutdown), get_max_node_jobs)
        self.logger = logging.getLogger(__name__)
        self.job_tracker = job_tracker
        self.job_factory = job_factory
        self.model_manager = model_manager
        self.pipe_manager = pipe_manager
        self.cancel_protocol = CancelProtocol(
            pipe_manager.router_pipes.router,
            self.job_queue,
            self.job_tracker
        )
        self.cache_protocol = CacheProtocol(
            pipe_manager.router_pipes.router,
            self.job_tracker,
            self.job_queue,
            self._rebuild_job
        )
        
        Thread(target=self._job_runner_loop, args=()).start()

    def _job_runner_loop(self):
        """Main job processing loop using FSM."""
        try:
            while True:
                network_job = self.job_queue.wait_for_job()
                if network_job is None:
                    return
                self._process_network_job(network_job)
        except Exception as e:
            self.logger.exception(f"Job runner loop failed: {e}")
            Thread(target=self._job_runner_loop, args=()).start()

    def _process_network_job(self, network_job: NetworkJob):
        """Take one packet off the queue and run the FSM over it."""
        job = self.job_tracker.get_job(network_job.job_id)
        if job is None:
            if network_job.job_id in self.job_tracker.jobs_completed:
                return
            job = self._add_job(network_job)
            if job is None:
                return

        node_id = self.pipe_manager.router_pipes.router.node_id()

        # Validate network job
        if not job.receive_network_job(network_job, node_id):
            if job.passes.error is not None:
                self.cancel_protocol.cancel_job(job, job.passes.error)
            return

        pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
        if pipe is None:
            return

        end_model = self.model_manager.get_end_model(pipe.model_id)

        fsm = JobProcessor(JobContext(
            node_id=node_id,
            pipe=pipe,
            end_model=end_model,
            job=job,
            on_fail=self.cancel_protocol.cancel_job,
            prompt_cache=self.job_tracker.prompt_cache
        ))

        try:
            fsm.run()
        except Exception as e:
            self.logger.exception(f"Job processing failed: {e}")

    def _add_job(self, network_job: NetworkJob) -> Job | None:
        """Take on a job this node has not seen, answering the origin's tags."""
        pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
        assert pipe is not None
        node_id = self.pipe_manager.router_pipes.router.node_id()
        policy = CachePolicy(
            node_id,
            pipe,
            self.model_manager.get_end_model(pipe.model_id),
            self.job_tracker.prompt_cache
        )
        job, outcome = self.job_tracker.add_job(
            network_job,
            self.model_manager.get_config(pipe.model_id),
            policy,
            pipe.model_id,
        )
        if outcome == CacheOutcome.MISS:
            self.cache_protocol.send_cache_status(network_job.origin_node_id, CacheStatus(
                network_job.job_id, network_job.pipe_id,
                network_job.attempt, CacheReason.MISS
            ))
            return None
        if outcome == CacheOutcome.NO_STORE:
            self.cache_protocol.send_cache_status(network_job.origin_node_id, CacheStatus(
                network_job.job_id, network_job.pipe_id,
                network_job.attempt, CacheReason.NO_STORE
            ))
        assert job is not None
        return job

    def _rebuild_job(self, job: Job):
        """Run the job again from token 0, against fresh caches everywhere.

        Distinct from the replay that answers a failed hash: a replay resends
        one pass against caches that are still right, a rebuild throws every
        cache on the pipe away because one node's is missing. The nodes are told
        so they can free theirs now, but `attempt` on the next packet is what
        makes it safe - the ABORT can lose the race against the retry.
        """
        pipe = self.pipe_manager.get_pipe_by_pipe_id(job.pipe_id)
        if pipe is None:
            return
        dead_attempt = job.passes.attempt
        self.logger.info(
            f"Job {job.job_id[:4]} restarting uncached: "
            "a node on the pipe does not hold the prefix"
        )
        self.job_queue.drop_queued(job.job_id)
        job.rebuild()
        # The rest of the job stores nothing, so the budget it was holding for
        # what it would store is better spent on entries that still exist.
        if self.job_tracker.prompt_cache is not None and job.caching.reserved:
            self.job_tracker.prompt_cache.release(job.job_id)
            job.caching.reserved = False

        node_id = self.pipe_manager.router_pipes.router.node_id()
        for segment_node_id in {s.node_id for s in pipe.segments} - {node_id}:
            self.cache_protocol.send_cache_status(segment_node_id, CacheStatus(
                job.job_id, job.pipe_id, dead_attempt, CacheReason.ABORT
            ))

        fsm = JobProcessor(JobContext(
            node_id=node_id,
            pipe=pipe,
            end_model=self.model_manager.get_end_model(pipe.model_id),
            job=job,
            on_fail=self.cancel_protocol.cancel_job,
            prompt_cache=self.job_tracker.prompt_cache
        ))
        try:
            fsm.run()
        except Exception as e:
            self.logger.exception(f"Job rebuild failed: {e}")

    def restart_token(self, network_job: NetworkJob):
        """Send a packet that failed its hash back to the origin for a resend.

        `pass_idx` stays as it came in: it tells the origin which pass to send
        again, and the nodes that already computed that pass replay their saved
        output instead of running their layers a second time.
        """
        network_job.data = None
        network_job.data_hash = b''
        network_job.compute_step = ComputeStep.EMBED
        network_job.current_layer = 0
        pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
        if pipe is None:
            return
        pipe.send_job(network_job, network_job.origin_node_id)

    def receive_data(self, node_id: str, data: bytes):
        """Receive and validate incoming job data."""
        try:
            job, valid = NetworkJob.from_bytes(data)
        except Exception:
            return
        if not valid:
            self.restart_token(job)
            return

        self.job_queue.add_to_queue(node_id, job)