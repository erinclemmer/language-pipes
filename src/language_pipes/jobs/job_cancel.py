import logging

from distributed_state_network.handler import DSNodeServer

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_queue import JobQueue
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.util.byte_helper import ByteHelper

CANCEL_PROTOCOL = 2

class JobCancel:
    """Tells the node that owns a job to stop waiting on it.

    Sent when the pipe a job is running on can no longer carry it - a model was
    unloaded, or the segment for the next layer left the network. Without it the
    origin sits on the job until the stale timeout fires.
    """
    job_id: str
    pipe_id: str
    reason: str

    def __init__(self, job_id: str, pipe_id: str, reason: str):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.reason = reason

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.reason)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> 'JobCancel':
        bts = ByteHelper(data)
        return JobCancel(
            job_id=bts.read_string(),
            pipe_id=bts.read_string(),
            reason=bts.read_string()
        )

class CancelProtocol:
    _router: DSNodeServer
    _logger: logging.Logger

    def __init__(
        self, 
        router: DSNodeServer,
        job_queue: JobQueue,
        job_tracker: JobTracker
    ):
        self._router = router
        self._logger = logging.getLogger(__name__)
        self._job_queue = job_queue
        self._job_tracker = job_tracker

    def _send_cancel(self, node_id: str, cancel: JobCancel):
        bts = ByteHelper()
        bts.write_int(CANCEL_PROTOCOL)
        bts.write_bytes(cancel.to_bytes())
        data = bts.get_bytes()
        router = self._router
        try:
            if node_id == router.node_id():
                router.receive_data(data)
            else:
                router.send_to_node(node_id, data)
        except Exception as e:
            self._logger.warning(f"Could not send cancel for job {cancel.job_id[:4]} to {node_id}: {e}")

    def cancel_job(self, job: Job, reason: str):
        self._job_queue.drop_queued(job.job_id)
        origin_node_id = job.origin_node_id
        self._job_tracker.cancel_job(job, reason)
        if origin_node_id != self._router.config.node_id:
            self._send_cancel(origin_node_id, JobCancel(job.job_id, job.pipe_id, reason))

    def cancel_jobs(self, jobs: list[Job], reason: str):
        for job in jobs:
            self.cancel_job(job, reason)

    def cancel_pipe_jobs(self, pipe_ids: list[str], reason: str):
        self.cancel_jobs(self._job_tracker.jobs_for_pipes(pipe_ids), reason)

    def cancel_model_jobs(self, model_id: str, reason: str):
        self.cancel_jobs(self._job_tracker.jobs_for_model(model_id, self._router.config.node_id), reason)

    def receive_cancel(self, data: bytes):
        """Handle a cancel sent by another node holding part of our job."""
        try:
            cancel = JobCancel.from_bytes(data)
        except Exception:
            return
        job = self._job_tracker.get_job(cancel.job_id)
        if job is None or job.pipe_id != cancel.pipe_id:
            self._job_queue.drop_queued(cancel.job_id)
            return
        self.cancel_job(job, cancel.reason)
