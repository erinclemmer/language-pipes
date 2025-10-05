from time import sleep
from threading import Thread
from typing import Callable, Optional, List
from distributed_state_network import DSNode

from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.job import Job
from language_pipes.llm_model.end_model import EndModel
from language_pipes.job_manager.enums import ComputeStep, JobStatus
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
    get_end_model: Callable[[str], Optional[EndModel]]
    restart_job: Callable[[Job], None]

    def __init__(
            self, 
            config: ProcessorConfig,
            router: DSNode,
            get_pipe: Callable[[str], Optional[Pipe]],
            get_end_model: Callable[[str], Optional[EndModel]],
            restart_job: Callable[[Job], None]
    ):
        self.router = router
        self.get_pipe = get_pipe
        self.get_end_model = get_end_model
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
            end_model = self.get_end_model(job.model_id)
            self.pending_jobs.pop()

            if pipe is None or not pipe.is_complete():
                self.restart_job(job)
                continue
            match job.current_step:
                case ComputeStep.LAYER:
                    model_for_job = pipe.model_for_job(job)
                    if model_for_job.virtual:
                        pipe.send_job(job, model_for_job.router_id)
                        continue
                    model_for_job.process_job(job)
                    if job.current_step == ComputeStep.LAYER:
                        model_for_job = pipe.model_for_job(job)
                        pipe.send_job(job, model_for_job.router_id)
                    else:
                        pipe.send_job(job, job.from_router_id)
                    continue
                case ComputeStep.NORM:
                    end_model.compute_norm(job)
                    end_model.compute_head(job)
                    if job.status == JobStatus.COMPLETED:
                        end_model.set_result(job)
                        pipe.complete_job(job)
                    else:
                        end_model.compute_embed(job)
                        send_to = pipe.model_for_job(job).router_id
                        pipe.send_job(job, send_to)
                    continue

    def receive_data(self, data: bytes):
        job = Job.from_bytes(data)
        job_certificate = self.router.cred_manager.read_public(job.from_router_id)
        if self.ecdsa_verification and not job.verify_signature(job_certificate):
            return

        for j in self.pending_jobs:
            if j.job_id == job.job_id:
                return
        self.pending_jobs.insert(0, job)

    def stop(self):
        self.httpd.stop()
        stop_thread(self.thread)