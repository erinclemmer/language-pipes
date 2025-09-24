from time import sleep
from threading import Thread
from typing import Callable, Optional, List
from distributed_state_network import DSNode

from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.job import Job
from language_pipes.job_manager.enums import JobStatus, ComputeStep
from language_pipes.handlers.job import JobServer
from language_pipes.util import stop_thread
from language_pipes.config.processor import ProcessorConfig

class JobReceiver:
    port: int
    public_key_file: str
    private_key_file: str
    ecdsa_verification: bool
    router: DSNode
    pending_jobs: List[Job]
    get_pipe: Callable[[str], Optional[Pipe]]
    restart_job: Callable[[Job], None]

    def __init__(
            self, 
            config: ProcessorConfig,
            router: DSNode,
            get_pipe: Callable[[str], Optional[Pipe]],
            restart_job: Callable[[Job], None]
    ):
        self.router = router
        self.get_pipe = get_pipe
        self.restart_job = restart_job
        self.pending_jobs = []
        self.ecdsa_verification = config.ecdsa_verification

        public_key = router.cert_manager.public_path(router.config.node_id)
        if public_key is None:
            msg = f"Could not find public key for self"
            router.logger.exception(msg)
            raise Exception(msg)

        thread, httpd = JobServer.start(config.https, public_key, config.job_port, self.router, self.receive_data)
        self.thread = thread
        self.httpd = httpd
        Thread(target=self.job_runner, args=()).start()
        router.logger.info(f"Started Job Receiver on port {config.job_port}")

    def job_runner(self):
        while True:
            if self.router.shutting_down:
                return
            if len(self.pending_jobs) == 0:
                sleep(0.1)
                continue
            
            job = self.pending_jobs[-1]
            pipe = self.get_pipe(job.pipe_id)
            if pipe is None or not pipe.is_complete():
                self.restart_job(job)
                return
            
            if job.current_step == ComputeStep.TOKENIZE and job.from_router_id != job.router_id:
                return
            
            if job.current_step != ComputeStep.TOKENIZE and job.status != JobStatus.COMPLETED:
                if job.from_router_id not in pipe.peers():
                    return
            self.pending_jobs.pop()
            pipe.process_job(job)

    def receive_data(self, data: bytes):
        job = Job.from_bytes(data)
        job_certificate = self.router.cred_manager.read_public(job.from_router_id)
        if self.ecdsa_verification and not job.verify_signature(job_certificate):
            return

        self.pending_jobs.insert(0, job)

    def stop(self):
        self.httpd.stop()
        stop_thread(self.thread)
