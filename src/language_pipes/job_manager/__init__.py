import gc
import ctypes
import random
import requests
from time import time, sleep
from typing import List, Optional, Tuple, Callable

import torch
from promise import Promise

from uuid import uuid4
from threading import Thread
from distributed_state_network import DSNode

from language_pipes.util.meta import MetaPipe
from language_pipes.llm_model.end_model import EndModel
from language_pipes.job_manager.router_pipes import RouterPipes
from language_pipes.job_manager.job import Job
from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.enums import JobStatus
from language_pipes.job_manager.layer_job import LayerTime, LayerJob
from language_pipes.util.chat import ChatMessage
from language_pipes.config.processor import ProcessorConfig
from language_pipes.llm_model import LlmModel
from language_pipes.llm_model.computed import validate_model
from language_pipes.job_manager.pending_job import PendingJob

CHECK_JOB_INTERVAL = 10
EXPIRED_JOB_TIME = 60  # Unified timeout for both prefill and decode phases

try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except:
    _malloc_trim = None

class JobManager:
    completed_jobs: List[str]
    jobs_pending: List[PendingJob]
    models: List[LlmModel]
    end_models: List[EndModel]
    app_dir: str
    
    router: DSNode
    
    router_pipes: RouterPipes
    config: ProcessorConfig
    started: bool

    def __init__(self, app_dir: str, router: DSNode, config: ProcessorConfig):
        self.started = False
        self.router = router
        self.config = config
        self.app_dir = app_dir
        self.logger = self.router.logger

        self.jobs_pending = []
        self.completed_jobs = []
        self.models = []
        self.end_models = []
        self.pipes_hosted = []
        self.router_pipes = RouterPipes(router)
        self.router.update_data("job_port", str(self.config.job_port))
        for m in self.config.hosted_models:
            self.host_model(m.id, m.max_memory, m.device, m.load_ends)
        
        self.print_pipes()

        self.started = True
        Thread(target=self.check_stale_jobs, args=( )).start()

    def check_stale_jobs(self):
        while True:
            remove_jobs = []
            for j in self.jobs_pending:
                stale_time = time() - j.last_update
                # Unified timeout - prefill chunks regularly update last_update,
                # so both prefill and decode phases use the same timeout
                if stale_time > EXPIRED_JOB_TIME:
                    self.logger.warning(
                        f"[Stale] job={j.job.job_id[:8]} timed out after {stale_time:.1f}s "
                        f"(token={j.job.current_token})"
                    )
                    remove_jobs.append(j.job.job_id)

            if len(remove_jobs) == 0:
                sleep(CHECK_JOB_INTERVAL)
                continue
        
            for job_id in remove_jobs:
                self.jobs_pending = [j for j in self.jobs_pending if j.job.job_id != job_id]
            
            gc.collect()
            torch.cuda.empty_cache()
            if _malloc_trim is not None:
                _malloc_trim(0)

            sleep(CHECK_JOB_INTERVAL)

    def print_pipes(self):
        for p in self.router_pipes.network_pipes():
            p.print(self.logger)

    def raise_exception(self, msg: str):
        self.logger.exception(msg)
        raise Exception(msg)

    def stop(self):
        for m in self.models:
            m.cleanup_tensors()
        for m in self.end_models:
            m.clean_up()
        self.models = []
        self.end_models = []
    
    def get_pipe(self, pipe_id: str) -> Optional[Pipe]:
        meta_pipe = self.router_pipes.network_pipe(pipe_id)
        if meta_pipe is None:
            return None
        return Pipe.from_meta(
            meta_pipe=meta_pipe,
            hosted_models=self.models,
            router=self.router,
            app_dir=self.app_dir,
            get_job_port=self.get_job_port,
            complete_job=self.complete_job,
            update_job=self.update_job,
            restart_job=self.restart_job
        )
    
    def get_job_port(self, router_id: str) -> Optional[int]:
        try:
            return int(self.router.read_data(router_id, 'job_port'))
        except Exception as e:
            self.logger.exception("Error getting job port: %s", e)
            return None

    def update_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.completed_jobs:
            return
        pending_job = self.get_pending_job(job_id)
        if pending_job is None:
            return
        self.logger.info(f'Received job update for {job_id}\n')
        pending_job.last_update = time()
        return pending_job.update(job)

    def complete_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.completed_jobs:
            return
        self.completed_jobs.append(job_id)
        pending_job = self.get_pending_job(job_id)
        if pending_job is None:
            return
        self.logger.info(f'Received job complete for {job_id}\n')
        pending_job.resolve(job)
        self.jobs_pending = [j for j in self.jobs_pending if j.job.job_id != job_id]
      
    def get_model_for_pipe(self, model_id: str, pipe: MetaPipe, device: str, available_memory: int) -> Tuple[int, Optional[LlmModel]]:
        start_memory = available_memory

        new_model: Optional[LlmModel] = LlmModel.from_id(self.app_dir, model_id, self.router.config.node_id, pipe.pipe_id, device)
        computed = new_model.computed
        if self.config.model_validation and len(pipe.segments) > 0 and not validate_model(new_model.computed.to_meta(), pipe.get_computed()):
            self.logger.warning(f'Computed data for model {model_id} does not match')
            return available_memory, None
        
        num_layers_to_load = int(available_memory // computed.avg_layer_size) - 1
        total_layers = new_model.collector.config.num_hidden_layers
        start_layer = pipe.next_start_layer()
        if num_layers_to_load == -1:
            start_layer = -1
            end_layer = -1
        else:
            end_layer = min([start_layer + num_layers_to_load, pipe.next_end_layer(total_layers), new_model.num_hidden_layers]) if start_layer != -1 else -1
            available_memory = available_memory - (end_layer - start_layer + 1) * computed.avg_layer_size

        if num_layers_to_load > -1 and end_layer != -1 and start_layer != -1:
            self.logger.info(f'Using {(start_memory - available_memory) / 10**9:.2f} GB of memory to load model {model_id}')
            new_model.start_layer = start_layer
            new_model.end_layer = end_layer
            new_model.input_embedding = None
            new_model.head = None
            new_model.print()
        else:
            new_model = None
        return available_memory, new_model

    def update_job_time(self, job_id: str):
        """Update the last_update time for a pending job to prevent stale timeout."""
        pending_job = self.get_pending_job(job_id)
        if pending_job is None:
            return
        pending_job.last_update = time()

    def add_pending_job(self, layer_job: LayerJob):
        existing = self.get_pending_job(layer_job.job_id)
        if existing is not None:
            return existing  # Return existing job instead of None
        job = Job(
            self.router.config.node_id, 
            layer_job.origin_node_id, 
            0, [], layer_job.pipe_id, ""
        )
        job.job_id = layer_job.job_id
        job.prompt_tokens = layer_job.data.state.size()[1]
        pending_job = PendingJob(job, time(), None, None)
        self.jobs_pending.append(pending_job)
        return pending_job

    def get_pending_job(self, job_id: str) -> Optional[Job]:
        for j in self.jobs_pending:
            if j.job.job_id == job_id:
                return j
        return None

    def load_end_model(self, model_id: str, device: int):
        model = EndModel(self.app_dir, model_id, device)
        self.end_models.append(model)
        return model

    def host_model(self, model_id: str, max_memory: float, device: str, load_ends: bool):
        available_memory = max_memory * 10 ** 9
        models_to_load: List[LlmModel] = []
        end_model = None
        if load_ends:
            end_model = self.load_end_model(model_id, device)
        
        for pipe_id in [p.pipe_id for p in self.router_pipes.pipes_for_model(model_id, False)]:
            if pipe_id not in self.pipes_hosted and len(self.pipes_hosted) >= self.config.max_pipes:
                break
            loaded = True
            while loaded:
                pipe = self.router_pipes.network_pipe(pipe_id)
                if pipe is None: 
                    break
                available_memory, model = self.get_model_for_pipe(model_id, pipe, device, available_memory)
                loaded = model is not None
                if model is not None:
                    self.pipes_hosted.append(model.pipe_id)
                    self.router_pipes.add_model_to_network(model.to_meta())
                    models_to_load.append(model)

        if len(self.pipes_hosted) < self.config.max_pipes:
            new_pipe = MetaPipe(str(uuid4()), model_id, [])
            self.pipes_hosted.append(new_pipe.pipe_id)
            _, model = self.get_model_for_pipe(model_id, new_pipe, device, available_memory)
            if model is not None:
                self.router_pipes.add_model_to_network(model.to_meta())
                models_to_load.append(model)

        if load_ends:
            end_model.load()

        for m in models_to_load:
            m.load()
            self.router_pipes.update_model(m.to_meta())
            self.models.append(m)

    def get_job_pipe(self, model_id: str) -> Optional[MetaPipe]:
        available_pipes: List[MetaPipe] = []
        for p in self.router_pipes.pipes_for_model(model_id, True):
            if not p.is_loading():
                available_pipes.append(p)
        if len(available_pipes) == 0:
            return None

        return random.choice(available_pipes)
    
    def get_end_model(self, model_id: str) -> Optional[EndModel]:
        for m in self.end_models:
            if m.model_id == model_id:
                return m
        return None

    def restart_job(self, job: Job):
        pipe = self.get_job_pipe(job.model_id)
        if pipe is None:
            job.status = JobStatus.ERROR
            ip = self.router.connection_from_node(job.router_id).address
            port = self.get_job_port(job.router_id)
            cert = self.router.cert_manager.public_path(job.router_id)
            requests.post(f"https://{ip}:{port}", data=job.to_bytes(), headers={ 'Content-Type': 'application/octet-stream' }, verify = cert)
            return
        self.start_job(
            job.model_id, 
            job.messages, 
            job.tokens, 
            pipe_id=pipe.pipe_id, 
            job_id=job.job_id
        )

    def start_job(
        self, 
        model_id: str, 
        messages: List[ChatMessage], 
        tokens: int, 
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        start: Optional[Callable] = None,
        update: Optional[Callable] = None,
        resolve: Optional[Promise] = None,
        pipe_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> Optional[Job]:
        end_model = self.get_end_model(model_id)
        if end_model is None:
            if resolve is not None:
                resolve('NO_ENDS')
                return None
            self.raise_exception(f"Could not find local end model for {model_id}")

        if pipe_id is None:
            network_pipe = self.get_job_pipe(model_id)
            if network_pipe is None:
                if resolve is not None:
                    resolve('NO_PIPE')
                    return None
                self.raise_exception(f"Could not find pipe for {model_id}")
            pipe_id = network_pipe.pipe_id

        pipe = self.get_pipe(pipe_id)
        if pipe is None:
            self.raise_exception(f"Could not find pipe {pipe_id}")

        job = Job(self.router.config.node_id, self.router.config.node_id, tokens, messages, pipe_id, model_id, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, presence_penalty=presence_penalty)
        
        if job_id is not None:
            job.job_id = job_id
        
        lt = LayerTime(
            node_id=self.router.config.node_id,
            is_embed=True
        )
        
        # Tokenize first to get prompt length
        end_model.tokenize(job)
        
        # Create pending job with chunking initialization
        pending_job = PendingJob(
            job, time(), resolve, update,
            prompt_length=job.prompt_tokens,
            chunk_size=self.config.prefill_chunk_size
        )
        
        # Set prefill start time
        job.prefill_start_time = time()
        job.chunk_start_time = job.prefill_start_time
        
        # Log prefill start
        if pending_job.chunking.is_active():
            self.logger.info(
                f"[Prefill] job={job.job_id[:8]} started: "
                f"prompt_tokens={job.prompt_tokens}, "
                f"chunks={pending_job.chunking.total_chunks}, "
                f"chunk_size={pending_job.chunking.chunk_size}"
            )
        else:
            self.logger.info(
                f"[Prefill] job={job.job_id[:8]} started: "
                f"prompt_tokens={job.prompt_tokens} (no chunking)"
            )
        
        # Get chunk range and compute embed for first chunk
        chunk_start, chunk_end = pending_job.chunking.get_range()
        
        # Log first chunk being processed
        if pending_job.chunking.is_active():
            self.logger.info(
                f"[Prefill] job={job.job_id[:8]} chunk 1/{pending_job.chunking.total_chunks} "
                f"starting: tokens {chunk_start}-{chunk_end}"
            )
        
        end_model.compute_embed(job, pending_job.cache, chunk_start, chunk_end)
        first_layer_model = pipe.model_for_job(job)
        if first_layer_model is None:
            self.raise_exception("Could not find appropriate model for processing")
            return None
        
        lt.send_time = time()
        layer_job = job.to_layer_job()

        if self.config.print_job_data:
            job.print_job(self.router.logger)
        
        layer_job.times.append(lt)
        pipe.send_job(layer_job, first_layer_model.node_id)
        self.jobs_pending.append(pending_job)

        if start is not None:
            start(job)

        return job
