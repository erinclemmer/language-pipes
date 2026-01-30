import random
from time import sleep
from threading import Thread
from typing import Callable, Optional, List

from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.job import ComputeStep
from language_pipes.jobs.job_factory import JobFactory
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.job_processor import JobProcessor, JobContext

from language_pipes.modeling.model_manager import ModelManager

from language_pipes.config import LpConfig

class JobReceiver:
    config: LpConfig
    job_factory: JobFactory
    job_queue: List[NetworkJob]
    pipe_manager: PipeManager
    model_manager: ModelManager
    is_shutdown: Callable[[], bool]

    def __init__(
            self, 
            config: LpConfig,
            logger,
            job_factory: JobFactory,
            job_tracker: JobTracker,
            pipe_manager: PipeManager,
            model_manager: ModelManager,
            is_shutdown: Callable[[], bool]
    ):
        self.job_queue = []
        self.job_tracker = job_tracker
        self.job_factory = job_factory
        self.model_manager = model_manager
        self.pipe_manager = pipe_manager
        self.config = config
        self.is_shutdown = is_shutdown
        self.logger = logger

        Thread(target=self._job_runner_loop, args=()).start()

    def _wait_for_job(self) -> Optional[NetworkJob]:
        """Wait for a job from the queue. Returns None if shutting down."""
        while True:
            if self.is_shutdown():
                return None
            if len(self.job_queue) > 0:
                idx = random.randrange(len(self.job_queue))
                network_job = self.job_queue.pop(idx)
                return network_job
            sleep(0.01)

    def _job_runner_loop(self):
        """Main job processing loop using FSM."""
        network_job = self._wait_for_job()
        if network_job is None:
            return
        
        job = self.job_tracker.get_job(network_job.job_id)
        if job is None:
            job = self.job_tracker.add_job(network_job)

        # Validate network job
        if not job.receive_network_job(network_job):
            return

        pipe = self.pipe_manager.get_pipe_by_pipe_id(network_job.pipe_id)
        if pipe is None:
            return

        end_model = self.model_manager.get_end_model(pipe.model_id)
        
        fsm = JobProcessor(JobContext(
            logger=self.logger,
            pipe=pipe,
            end_model=end_model,
            job=job,
            config=self.config
        ))

        try:
            fsm.run()
        except Exception as e:
            print(e)
        
        Thread(target=self._job_runner_loop, args=()).start()

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

    def receive_data(self, data: bytes):
        """Receive and validate incoming job data."""
        job, valid = NetworkJob.from_bytes(data)
        if not valid:
            self.restart_job(job)
            return
        
        # Ignore duplicate jobs
        for j in self.job_queue:
            if j.job_id == job.job_id:
                return

        self.job_queue.insert(0, job)
