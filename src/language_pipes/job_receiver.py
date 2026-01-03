from time import sleep, time
from threading import Thread
from typing import Callable, Optional, List
from distributed_state_network import DSNode

from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.job import Job
from language_pipes.llm_model.end_model import EndModel
from language_pipes.job_manager.enums import ComputeStep, JobStatus
from language_pipes.handlers.job import JobServer
from language_pipes.util import stop_thread
from language_pipes.job_manager.layer_job import LayerJob, LayerTime
from language_pipes.config.processor import ProcessorConfig
from language_pipes.job_manager.pending_job import PendingJob

class JobReceiver:
    port: int
    public_key_file: str
    private_key_file: str
    ecdsa_verification: bool
    print_times: bool
    print_job_data: bool
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
        self.print_times = config.print_times
        self.print_job_data = config.print_job_data

        thread, httpd = JobServer.start(config.job_port, self.router, self.receive_data)
        self.thread = thread
        self.httpd = httpd
        Thread(target=self.job_runner, args=()).start()
        router.logger.info(f"Started Job Receiver on port {config.job_port}")

    def job_runner(self):        
        try:
            while True:
                if self.router.shutting_down:
                    return
                
                if len(self.pending_jobs) == 0:
                    sleep(0.1)
                else:
                    break
            
            layer_job = self.pending_jobs[-1]
            self.pending_jobs.pop()
            
            pipe = self.get_pipe(layer_job.pipe_id)
            end_model = self.get_end_model(pipe.model_id)

            if pipe is None or not pipe.is_complete():
                job = self.get_pending_job(layer_job.job_id).job
                self.restart_job(job)
                return Thread(target=self.job_runner, args=()).start()
            
            if layer_job.restart:
                job = self.get_pending_job(layer_job.job_id).job
                job.current_step = ComputeStep.EMBED
                lt = LayerTime(
                    node_id=self.router.config.node_id,
                    is_embed=True
                )
                end_model.compute_embed(job)
                lt.send_time = time()
                layer_job = job.to_layer_job()
                layer_job.times.append(lt)
                model = pipe.model_for_job(layer_job)
                if model.virtual:
                    pipe.send_job(layer_job, model.node_id)
                    return

            if layer_job.done:
                job = self.get_pending_job(layer_job.job_id).job
                job.current_step = ComputeStep.NORM
                job.data = layer_job.data

                lt = LayerTime(
                    node_id=self.router.config.node_id,
                    is_head=True
                )
                end_model.compute_norm(job)
                end_model.compute_head(job)
                lt.send_time = time()
                layer_job.times.append(lt)
                if self.print_times:
                    layer_job.print_times(self.router.logger)
                if self.print_job_data:
                    job.print_job(self.router.logger)
                layer_job.times = []
                
                if job.status == JobStatus.COMPLETED:
                    end_model.set_result(job)
                    pipe.complete_job(job)
                    return Thread(target=self.job_runner, args=()).start()
                else:
                    if not pipe.update_job(job):
                        return Thread(target=self.job_runner, args=()).start()
                    lt = LayerTime(
                        node_id=self.router.config.node_id,
                        is_embed=True
                    )
                    end_model.compute_embed(job)
                    lt.send_time = time()
                    layer_job = job.to_layer_job()
                    layer_job.times.append(lt)

            model = pipe.model_for_job(layer_job)
            model.process_job(layer_job)

            if layer_job.done:
                pipe.send_job(layer_job, layer_job.origin_node_id)
            else:
                model = pipe.model_for_job(layer_job)
                pipe.send_job(layer_job, model.node_id)
        except Exception as e:
            print(e)
        Thread(target=self.job_runner, args=()).start()

    def restart_token(self, job: LayerJob):
        job.restart = True
        pipe = self.get_pipe(job.pipe_id)
        pipe.send_job(job, job.origin_node_id)

    def receive_data(self, data: bytes):
        job = LayerJob.from_bytes(data)
        
        for j in self.pending_jobs:
            if j.job_id == job.job_id:
                return

        valid = job.data.validate_state(job.data_hash)
        if not job.restart and (valid is False or valid is None):
            self.restart_token(job)
            return

        self.pending_jobs.insert(0, job)

    def stop(self):
        self.httpd.stop()
        stop_thread(self.thread)
