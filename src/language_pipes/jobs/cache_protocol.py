import logging
from typing import Callable

from distributed_state_network.handler import DSNodeServer

from language_pipes.jobs.cache_packets import CacheReason, CacheStatus
from language_pipes.jobs.job import Job
from language_pipes.jobs.job_queue import JobQueue
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.util.byte_helper import ByteHelper

CACHE_PROTOCOL = 3

class CacheProtocol:
    _router: DSNodeServer
    _logger: logging.Logger
    _job_tracker: JobTracker
    _job_queue: JobQueue

    def __init__(
        self,
        router: DSNodeServer,
        job_tracker: JobTracker,
        job_queue: JobQueue,
        rebuild_job: Callable[[Job], None]
    ):
        self._router = router
        self._logger = logging.getLogger(__name__)
        self._job_tracker = job_tracker
        self._job_queue = job_queue
        self._rebuild_job = rebuild_job

    def send_cache_status(self, node_id: str, status: CacheStatus):
        bts = ByteHelper()
        bts.write_int(CACHE_PROTOCOL)
        bts.write_bytes(status.to_bytes())
        data = bts.get_bytes()
        try:
            if node_id == self._router.node_id():
                self._router.receive_data(data)
            else:
                self._router.send_to_node(node_id, data)
        except Exception as e:
            self._logger.warning(
                f"Could not send cache status for job {status.job_id[:4]} to {node_id}: {e}"
            )

    def receive_cache_status(self, data: bytes):
        """Handle the prompt cache's back-channel (see `jobs/cache_packets.py`)."""
        try:
            status = CacheStatus.from_bytes(data)
        except Exception:
            return
        job = self._job_tracker.get_job(status.job_id)
        if job is None or job.pipe_id != status.pipe_id:
            return
        # An answer about an attempt that is already dead must not act on the
        # retry that replaced it, or a job could ping-pong between rebuilds.
        if status.attempt != job.passes.attempt:
            self._logger.debug(
                f"Job {job.job_id[:4]} ignoring {status.reason.name} "
                f"for attempt {status.attempt} (now {job.passes.attempt})"
            )
            return

        # `MISS` and `NO_STORE` are answers to a job this node dispatched;
        # `ABORT` is an instruction from the node that dispatched one to us.
        is_origin = job.origin_node_id == self._router.node_id()
        if status.reason == CacheReason.ABORT:
            if is_origin:
                return
            self._job_queue.drop_queued(job.job_id)
            self._job_tracker.remove_job(job.job_id)
            return
        if not is_origin:
            return

        if status.reason == CacheReason.MISS:
            self._rebuild_job(job)
        elif status.reason == CacheReason.NO_STORE:
            job.caching.stop_writing()