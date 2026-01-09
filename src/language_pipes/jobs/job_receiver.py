from time import sleep, time
from threading import Thread
from typing import Callable, Optional, List, Tuple
from distributed_state_network import DSNode

from language_pipes.pipes.pipe import Pipe

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_handler import JobServer
from language_pipes.jobs.pending_job import PendingJob
from language_pipes.jobs.job_manager import JobManager
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.jobs.layer_job import LayerJob, LayerTime
from language_pipes.jobs.job_receiver_fsm import JobReceiverFSM

from language_pipes.modeling.end_model import EndModel

from language_pipes.util import stop_thread
from language_pipes.util.enums import ComputeStep, JobStatus

from language_pipes.config import LpConfig

class JobReceiver:
    router: DSNode
    print_times: bool
    print_job_data: bool
    job_manager: JobManager
    job_queue: List[LayerJob]
    get_end_model: Callable[[str], Optional[EndModel]]

    def __init__(
            self, 
            config: LpConfig,
            router: DSNode,
            job_manager: JobManager,
            job_tracker: JobTracker,
            get_end_model: Callable[[str], Optional[EndModel]]
    ):
        self.router = router
        self.get_end_model = get_end_model
        self.job_queue = []
        self.job_tracker = job_tracker
        self.job_manager = job_manager
        self.print_times = config.print_times
        self.print_job_data = config.print_job_data

        thread, httpd = JobServer.start(config.job_port, self.router, self.receive_data)
        self.thread = thread
        self.httpd = httpd
        Thread(target=self._job_runner_loop, args=()).start()
        router.logger.info(f"Started Job Receiver on port {config.job_port}")

    def _wait_for_job(self) -> Optional[LayerJob]:
        """Wait for a job from the queue. Returns None if shutting down."""
        while True:
            if self.router.shutting_down:
                return None
            if len(self.job_queue) > 0:
                layer_job = self.job_queue.pop()
                return layer_job
            sleep(0.01)

    def _job_runner_loop(self):
        """Main job processing loop using FSM."""
        try:
            fsm = JobReceiverFSM(self)
            fsm.run()
        except Exception as e:
            print(e)
        
        Thread(target=self._job_runner_loop, args=()).start()

    def restart_token(self, job: LayerJob):
        """Mark job for restart and send back to origin."""
        job.restart = True
        pipe = self.job_manager.get_pipe(job.pipe_id)
        if pipe is None:
            return
        pipe.send_job(job, job.origin_node_id)

    def receive_data(self, data: bytes):
        """Receive and validate incoming job data."""
        job = LayerJob.from_bytes(data)
        
        # Ignore duplicate jobs
        for j in self.job_queue:
            if j.job_id == job.job_id:
                return

        # Validate state integrity
        valid = job.data.validate_state(job.data_hash)
        if not job.restart and (valid is False or valid is None):
            self.restart_token(job)
            return

        self.job_queue.insert(0, job)

    def stop(self):
        """Stop the job receiver."""
        self.httpd.stop()
        stop_thread(self.thread)
