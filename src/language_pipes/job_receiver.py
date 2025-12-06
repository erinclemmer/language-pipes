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
from language_pipes.job_manager.layer_job import LayerJob
from language_pipes.config.processor import ProcessorConfig
from language_pipes.job_manager.pending_job import PendingJob

class JobReceiver:
    port: int
    public_key_file: str
    private_key_file: str
    ecdsa_verification: bool
    router: DSNode
    pending_jobs: List[LayerJob]
    get_pipe: Callable[[str], Optional[Pipe]]
    get_end_model: Callable[[str], Optional[EndModel]]
    restart_job: Callable[[Job], None]

    def __init__(
            self, 
            config: ProcessorConfig,
            router: DSNode,
            get_pipe: Callable[[str], Optional[Pipe]],
            get_end_model: Callable[[str], Optional[EndModel]],
            get_pending_job: Callable[[str], Optional[PendingJob]],
            restart_job: Callable[[Job], None]
    ):
        self.router = router
        self.get_pipe = get_pipe
        self.get_end_model = get_end_model
        self.restart_job = restart_job
        self.get_pending_job = get_pending_job
        self.pending_jobs = []
        self.ecdsa_verification = config.ecdsa_verification

        thread, httpd = JobServer.start(config.job_port, self.router, self.receive_data)
        self.thread = thread
        self.httpd = httpd
        Thread(target=self.job_runner, args=()).start()
        router.logger.info(f"Started Job Receiver on port {config.job_port}")

    def job_runner(self):
        while True:
            try:
                if self.router.shutting_down:
                    return
                
                if len(self.pending_jobs) == 0:
                    sleep(0.1)
                    continue
                
                layer_job = self.pending_jobs[-1]
                self.pending_jobs.pop()
                
                pipe = self.get_pipe(layer_job.pipe_id)
                end_model = self.get_end_model(pipe.model_id)

                if pipe is None or not pipe.is_complete():
                    self.restart_job(layer_job)
                    continue
                
                if layer_job.done:
                    job = self.get_pending_job(layer_job.job_id).job
                    job.current_step = ComputeStep.NORM
                    job.data = layer_job.data
                    end_model.compute_norm(job)
                    end_model.compute_head(job)
                    if job.status == JobStatus.COMPLETED:
                        end_model.set_result(job)
                        pipe.complete_job(job)
                        continue
                    else:
                        if not pipe.update_job(job):
                            continue
                        end_model.compute_embed(job)
                        layer_job = job.to_layer_job()

                model = pipe.model_for_job(layer_job)
                model.process_job(layer_job)

                if layer_job.done:
                    pipe.send_job(layer_job, layer_job.origin_node_id)
                else:
                    model = pipe.model_for_job(layer_job)
                    pipe.send_job(layer_job, model.router_id)
            except Exception as e:
                print(e)

    def receive_data(self, data: bytes):
        job = LayerJob.from_bytes(data)
        
        for j in self.pending_jobs:
            if j.job_id == job.job_id:
                return

        self.pending_jobs.insert(0, job)

    def stop(self):
        self.httpd.stop()
        stop_thread(self.thread)