import logging
import random
import threading
from time import sleep
from threading import Thread
from typing import Callable, Dict, Optional, List

from language_pipes.jobs.job_queue import JobQueue
from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.cache_packets import CacheReason, CacheStatus
from language_pipes.jobs.cache_policy import CacheOutcome, CachePolicy
from language_pipes.jobs.job import ComputeStep, Job
from language_pipes.jobs.job_cancel import JobCancel
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.jobs.job_processor import JobProcessor, JobContext
from language_pipes.util.byte_helper import ByteHelper

CANCEL_PROTOCOL = 2
CACHE_PROTOCOL = 3


class JobReceiver:
    job_factory: JobFactory
    job_queue: JobQueue
    pipe_manager: PipeManager
    model_manager: ModelManager
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
                self.cancel_job(job, job.passes.error)
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
            on_fail=self.cancel_job,
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
            self._send_cache_status(network_job.origin_node_id, CacheStatus(
                network_job.job_id, network_job.pipe_id,
                network_job.attempt, CacheReason.MISS
            ))
            return None
        if outcome == CacheOutcome.NO_STORE:
            self._send_cache_status(network_job.origin_node_id, CacheStatus(
                network_job.job_id, network_job.pipe_id,
                network_job.attempt, CacheReason.NO_STORE
            ))
        assert job is not None
        return job

    def _node_id(self) -> str:
        return self.pipe_manager.router_pipes.router.node_id()

    def _send_cancel(self, node_id: str, cancel: JobCancel):
        bts = ByteHelper()
        bts.write_int(CANCEL_PROTOCOL)
        bts.write_bytes(cancel.to_bytes())
        data = bts.get_bytes()
        router = self.pipe_manager.router_pipes.router
        try:
            if node_id == router.node_id():
                router.receive_data(data)
            else:
                router.send_to_node(node_id, data)
        except Exception as e:
            self.logger.warning(f"Could not send cancel for job {cancel.job_id[:4]} to {node_id}: {e}")

    def _send_cache_status(self, node_id: str, status: CacheStatus):
        bts = ByteHelper()
        bts.write_int(CACHE_PROTOCOL)
        bts.write_bytes(status.to_bytes())
        data = bts.get_bytes()
        router = self.pipe_manager.router_pipes.router
        try:
            if node_id == router.node_id():
                router.receive_data(data)
            else:
                router.send_to_node(node_id, data)
        except Exception as e:
            self.logger.warning(
                f"Could not send cache status for job {status.job_id[:4]} to {node_id}: {e}"
            )

    def receive_cache_status(self, data: bytes):
        """Handle the prompt cache's back-channel (see `jobs/cache_packets.py`)."""
        try:
            status = CacheStatus.from_bytes(data)
        except Exception:
            return
        job = self.job_tracker.get_job(status.job_id)
        if job is None or job.pipe_id != status.pipe_id:
            return
        # An answer about an attempt that is already dead must not act on the
        # retry that replaced it, or a job could ping-pong between rebuilds.
        if status.attempt != job.passes.attempt:
            self.logger.debug(
                f"Job {job.job_id[:4]} ignoring {status.reason.name} "
                f"for attempt {status.attempt} (now {job.passes.attempt})"
            )
            return

        # `MISS` and `NO_STORE` are answers to a job this node dispatched;
        # `ABORT` is an instruction from the node that dispatched one to us.
        is_origin = job.origin_node_id == self._node_id()
        if status.reason == CacheReason.ABORT:
            if is_origin:
                return
            self.job_queue.drop_queued(job.job_id)
            self.job_tracker.remove_job(job.job_id)
            return
        if not is_origin:
            return

        if status.reason == CacheReason.MISS:
            self._rebuild_job(job)
        elif status.reason == CacheReason.NO_STORE:
            job.caching.stop_writing()

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

        node_id = self._node_id()
        for segment_node_id in {s.node_id for s in pipe.segments} - {node_id}:
            self._send_cache_status(segment_node_id, CacheStatus(
                job.job_id, job.pipe_id, dead_attempt, CacheReason.ABORT
            ))

        fsm = JobProcessor(JobContext(
            node_id=node_id,
            pipe=pipe,
            end_model=self.model_manager.get_end_model(pipe.model_id),
            job=job,
            on_fail=self.cancel_job,
            prompt_cache=self.job_tracker.prompt_cache
        ))
        try:
            fsm.run()
        except Exception as e:
            self.logger.exception(f"Job rebuild failed: {e}")

    def cancel_job(self, job: Job, reason: str):
        """Stop a job here and, when it belongs to another node, upstream too.

        The origin node is the one holding the API request open, so it has to
        hear about the cancel; otherwise it waits out the stale timeout.
        """
        self.job_queue.drop_queued(job.job_id)
        origin_node_id = job.origin_node_id
        self.job_tracker.cancel_job(job, reason)
        if origin_node_id != self._node_id():
            self._send_cancel(origin_node_id, JobCancel(job.job_id, job.pipe_id, reason))

    def cancel_jobs(self, jobs: List[Job], reason: str):
        for job in jobs:
            self.cancel_job(job, reason)

    def cancel_pipe_jobs(self, pipe_ids: List[str], reason: str):
        """Cancel every job running on the given pipes (a segment went away)."""
        self.cancel_jobs(self.job_tracker.jobs_for_pipes(pipe_ids), reason)

    def cancel_model_jobs(self, model_id: str, reason: str):
        """Cancel jobs this node started for a model whose end model is gone.

        Jobs that originated elsewhere do not use our end model, so they are
        left alone - their own origin owns that decision.
        """
        self.cancel_jobs(self.job_tracker.jobs_for_model(model_id, self._node_id()), reason)

    def receive_cancel(self, node_id: str, data: bytes):
        """Handle a cancel sent by another node holding part of our job."""
        try:
            cancel = JobCancel.from_bytes(data)
        except Exception:
            return
        job = self.job_tracker.get_job(cancel.job_id)
        if job is None or job.pipe_id != cancel.pipe_id:
            self.job_queue.drop_queued(cancel.job_id)
            return
        self.cancel_job(job, cancel.reason)

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