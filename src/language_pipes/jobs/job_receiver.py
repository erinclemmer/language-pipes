import logging
import random
import threading
from time import sleep
from threading import Thread
from typing import Callable, Dict, Optional, List

from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.job import ComputeStep
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.modeling.model_manager import ModelManager
from language_pipes.jobs.job_processor import JobProcessor, JobContext
from language_pipes.util.config import get_max_node_jobs

class JobReceiver:
    job_factory: JobFactory
    job_queue: Dict[str, List[NetworkJob]]
    queue_lock: threading.Lock
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
            is_shutdown: Callable[[], bool]
    ):
        self.job_queue = { }
        self.queue_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.job_tracker = job_tracker
        self.job_factory = job_factory
        self.model_manager = model_manager
        self.pipe_manager = pipe_manager
        self.is_shutdown = is_shutdown
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
                    pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
                    assert pipe is not None
                    job = self.job_tracker.add_job(network_job, self.model_manager.get_config(pipe.model_id))
                    assert job is not None

                # Validate network job
                if not job.receive_network_job(network_job):
                    return

                pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
                if pipe is None:
                    return

                end_model = self.model_manager.get_end_model(pipe.model_id)
                
                fsm = JobProcessor(JobContext(
                    node_id=self.pipe_manager.router_pipes.router.node_id(),
                    pipe=pipe,
                    end_model=end_model,
                    job=job
                ))

                try:
                    fsm.run()
                except Exception:
                    self.logger.exception("Job processing failed")
        except Exception:
            self.logger.exception("Job runner loop failed")

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
            if len(self.job_queue[node_id]) > get_max_node_jobs():
                raise Exception("Maximum number of jobs for node reached")
            self.job_queue[node_id].insert(0, job)
