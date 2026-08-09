import logging
import random
import threading
from time import sleep
from threading import Thread
from typing import Callable, Dict, Optional, List

from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.job import ComputeStep, Job
from language_pipes.jobs.job_cancel import JobCancel
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.jobs.job_processor import JobProcessor, JobContext
from language_pipes.util.byte_helper import ByteHelper

CANCEL_PROTOCOL = 2

class JobReceiver:
    job_factory: JobFactory
    job_queue: Dict[str, List[NetworkJob]]
    queue_lock: threading.Lock
    pipe_manager: PipeManager
    model_manager: ModelManager
    shutdown: bool
    is_shutdown: Callable[[], bool]
    get_max_node_jobs: Callable[[], int]

    def __init__(
            self,
            job_factory: JobFactory,
            job_tracker: JobTracker,
            pipe_manager: PipeManager,
            model_manager: ModelManager,
            is_shutdown: Callable[[], bool],
            get_max_node_jobs: Callable[[], int]
    ):
        self.job_queue = { }
        self.queue_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.job_tracker = job_tracker
        self.job_factory = job_factory
        self.model_manager = model_manager
        self.pipe_manager = pipe_manager
        self.is_shutdown = is_shutdown
        self.get_max_node_jobs = get_max_node_jobs
        self.shutdown = False
        
        Thread(target=self._job_runner_loop, args=()).start()

    def _wait_for_job(self) -> Optional[NetworkJob]:
        """Wait for a job from the queue. Returns None if shutting down."""
        while True:
            if self.is_shutdown() or self.shutdown:
                return None
            if len(self.job_queue.keys()) > 0:
                with self.queue_lock:
                    node_id = random.choice(list(self.job_queue.keys()))
                    node_jobs = self.job_queue[node_id]
                    idx = random.randrange(len(node_jobs))
                    network_job = self.job_queue[node_id].pop(idx)
                    if len(self.job_queue[node_id]) == 0:
                        del self.job_queue[node_id]
                return network_job
            sleep(0.01)

    def _job_runner_loop(self):
        """Main job processing loop using FSM."""
        try:
            while True:
                network_job = self._wait_for_job()
                if network_job is None:
                    return
                
                job = self.job_tracker.get_job(network_job.job_id)
                if job is None:
                    # A job that already finished or was canceled must not be
                    # resurrected by a packet that was still in flight.
                    if network_job.job_id in self.job_tracker.jobs_completed:
                        continue
                    pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
                    assert pipe is not None
                    job = self.job_tracker.add_job(
                        network_job,
                        self.model_manager.get_config(pipe.model_id),
                        pipe.model_id
                    )
                    assert job is not None

                # Validate network job
                if not job.receive_network_job(network_job):
                    continue

                pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
                if pipe is None:
                    continue

                end_model = self.model_manager.get_end_model(pipe.model_id)
                
                fsm = JobProcessor(JobContext(
                    node_id=self.pipe_manager.router_pipes.router.node_id(),
                    pipe=pipe,
                    end_model=end_model,
                    job=job,
                    on_fail=self.cancel_job
                ))

                try:
                    fsm.run()
                except Exception as e:
                    self.logger.exception(f"Job processing failed: {e}")
        except Exception as e:
            self.logger.exception(f"Job runner loop failed: {e}")
            Thread(target=self._job_runner_loop, args=()).start()

    def _node_id(self) -> str:
        return self.pipe_manager.router_pipes.router.node_id()

    def _drop_queued(self, job_id: str):
        """Discard packets for a job that is no longer running."""
        with self.queue_lock:
            for node_id in list(self.job_queue.keys()):
                self.job_queue[node_id] = [j for j in self.job_queue[node_id] if j.job_id != job_id]
                if len(self.job_queue[node_id]) == 0:
                    del self.job_queue[node_id]

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

    def cancel_job(self, job: Job, reason: str):
        """Stop a job here and, when it belongs to another node, upstream too.

        The origin node is the one holding the API request open, so it has to
        hear about the cancel; otherwise it waits out the stale timeout.
        """
        self._drop_queued(job.job_id)
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
            self._drop_queued(cancel.job_id)
            return
        self.cancel_job(job, cancel.reason)

    def restart_token(self, network_job: NetworkJob):
        """Mark job for restart and send back to origin."""
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
        
        # Ignore duplicate jobs
        if node_id in self.job_queue:
            for j in self.job_queue[node_id]:
                if j.job_id == job.job_id:
                    return

        with self.queue_lock:
            if node_id not in self.job_queue:
                self.job_queue[node_id] = [ ]
            if len(self.job_queue[node_id]) > self.get_max_node_jobs():
                raise Exception("Maximum number of jobs for node reached")
            self.job_queue[node_id].insert(0, job)
