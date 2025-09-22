import random
import requests
from typing import List, Optional, Tuple

from promise import Promise

from uuid import uuid4
from distributed_state_network import DSNode

from language_pipes.util.meta import MetaPipe
from language_pipes.job_manager.router_pipes import RouterPipes
from language_pipes.job_manager.job import Job
from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.enums import JobStatus
from language_pipes.util.chat import ChatMessage
from language_pipes.config.processor import ProcessorConfig
from language_pipes.llm_model import LlmModel
from language_pipes.llm_model.computed import validate_model

class PendingJob:
    job_id: str
    promise: Promise

    def __init__(self, job_id: str, promise: Promise):
        self.job_id = job_id
        self.promise = promise

class JobManager:
    completed_jobs: List[str]
    jobs_pending: List[PendingJob]
    models: List[LlmModel]
    
    router: DSNode
    
    router_pipes: RouterPipes
    config: ProcessorConfig
    started: bool

    def __init__(self, router: DSNode, config: ProcessorConfig):
        self.started = False
        self.router = router
        self.config = config
        self.logger = self.router.logger

        self.jobs_pending = []
        self.completed_jobs = []
        self.models = []
        self.pipes_hosted = 0
        self.router_pipes = RouterPipes(router)
        self.router.update_data("job_port", str(self.config.job_port))
        for m in self.config.hosted_models:
            self.host_model(m.id, m.max_memory, m.device)
        
        self.started = True

    def raise_exception(self, msg: str):
        self.logger.exception(msg)
        raise Exception(msg)

    def stop(self):
        for m in self.models:
            m.cleanup_tensors()
        self.models = []
    
    def get_pipe(self, pipe_id: str) -> Optional[Pipe]:
        meta_pipe = self.router_pipes.network_pipe(pipe_id)
        if meta_pipe is None:
            return None
        return Pipe.from_meta(
            meta_pipe=meta_pipe,
            hosted_models=self.models,
            router=self.router,
            https=self.config.https,
            get_job_port=self.get_job_port,
            complete_job=self.complete_job,
            restart_job=self.restart_job
        )
    
    def get_job_port(self, router_id: str) -> Optional[int]:
        try:
            return int(self.router.read_data(router_id, 'job_port'))
        except Exception as e:
            self.logger.exception("Error getting job port: %s", e)
            return None

    def complete_job(self, job: Job):
        job_id = job.job_id
        if job_id in self.completed_jobs:
            return
        self.completed_jobs.append(job_id)
        matches = [j for j in self.jobs_pending if j.job_id == job_id]
        if len(matches) == 0:
            return
        self.logger.info(f'\nReceived job complete for {job_id}\n')
        matches[0].promise(job)
        self.jobs_pending = [j for j in self.jobs_pending if j.job_id != job_id]
      
    def get_model_for_pipe(self, model_id: str, pipe: MetaPipe, device: str, available_memory: int) -> Tuple[int, Optional[LlmModel]]:
        start_memory = available_memory

        new_model: Optional[LlmModel] = LlmModel.from_id(model_id, self.router.config.node_id, pipe.pipe_id, device)
        computed = new_model.computed
        if self.config.model_validation and len(pipe.segments) > 0 and not validate_model(new_model.computed.to_meta(), pipe.get_computed()):
            self.logger.warning(f'Computed data for model {model_id} does not match')
            return available_memory, None
        
        load_embed = False
        if pipe.get_embed() is None and computed.embed_size <= available_memory:
            available_memory -= computed.embed_size
            load_embed = True
        
        load_head = False
        if pipe.get_head() is None and computed.head_size <= available_memory:
            available_memory -= computed.head_size
            load_head = True

        num_layers_to_load = int(available_memory // computed.avg_layer_size) - 1
        start_layer = pipe.next_start_layer()
        if num_layers_to_load == -1:
            start_layer = -1
            end_layer = -1
        else:
            end_layer = min([start_layer + num_layers_to_load, pipe.next_end_layer(), new_model.num_hidden_layers]) if start_layer != -1 else -1
            available_memory = available_memory - (end_layer - start_layer + 1) * computed.avg_layer_size

        if load_embed or load_head or (num_layers_to_load > -1 and end_layer != -1 and start_layer != -1):
            self.logger.info(f'Using {(start_memory - available_memory) / 10**9:.2f} GB of memory to load model {model_id}')
            new_model.start_layer = start_layer
            new_model.end_layer = end_layer
            new_model.input_embedding = None if not load_embed else True
            new_model.head = None if not load_head else True
            new_model.print()
        else:
            new_model = None
        return available_memory, new_model

    def host_model(self, model_id: str, max_memory: float, device: str):
        available_memory = max_memory * 10 ** 9
        models_to_load: List[LlmModel] = []
        for pipe_id in [p.pipe_id for p in self.router_pipes.pipes_for_model(model_id, False)]:
            if self.pipes_hosted >= self.config.max_pipes:
                break
            loaded = True
            while loaded:
                pipe = self.router_pipes.network_pipe(pipe_id)
                if pipe is None: 
                    break
                available_memory, model = self.get_model_for_pipe(model_id, pipe, device, available_memory)
                loaded = model is not None
                if model is not None:
                    self.pipes_hosted += 1
                    self.router_pipes.add_model_to_network(model.to_meta())
                    models_to_load.append(model)

        if self.pipes_hosted < self.config.max_pipes:
            new_pipe = MetaPipe(str(uuid4()), model_id, [])
            _, model = self.get_model_for_pipe(model_id, new_pipe, device, available_memory)
            if model is not None:
                self.router_pipes.add_model_to_network(model.to_meta())
                models_to_load.append(model)

        for m in models_to_load:
            m.load()
            self.router_pipes.update_model(m.to_meta())
            self.models.append(m)

        for p in self.router_pipes.network_pipes():
            p.print(self.logger)

    def get_job_pipe(self, model_id: str) -> Optional[MetaPipe]:
        available_pipes: List[MetaPipe] = []
        for p in self.router_pipes.pipes_for_model(model_id, True):
            if not p.is_loading():
                available_pipes.append(p)
        if len(available_pipes) == 0:
            return None

        return random.choice(available_pipes)

    def restart_job(self, job: Job):
        pipe = self.get_job_pipe(job.model_id)
        if pipe is None:
            job.status = JobStatus.ERROR
            ip = self.router.connection_from_node(job.router_id).address
            port = self.get_job_port(job.router_id)
            cert = self.router.cert_manager.public_path(job.router_id)
            requests.post(f"https://{ip}:{port}", data=job.to_bytes(), headers={ 'Content-Type': 'application/octet-stream' }, verify = cert)
            return
        self.start_job(job.router_id, job.model_id, pipe.pipe_id, job.messages, job.tokens, job.job_id)

    def start_job(self, router_id: str, model_id: str, pipe_id: str, messages: List[ChatMessage], tokens: int, job_id: Optional[str] = None) -> str:
        pipe = self.get_pipe(pipe_id)
        job = Job(self.router.config.node_id, self.router.config.node_id, tokens, messages, pipe_id, model_id, None)
        if job_id is not None:
            job.job_id = job_id
        if pipe is None:
            self.raise_exception(f"Could not find pipe {pipe_id}")
        try:
            pipe.send_job(job, router_id)
        except Exception as e:
            self.restart_job(job)

        return job.job_id

    def complete(self, model_id: str, messages: List[ChatMessage], tokens: int, promise: Promise) -> Optional[str]:
        network_pipe = self.get_job_pipe(model_id)
        if network_pipe is None:
            promise('NO_PIPE')
            return None
        peers = network_pipe.peers()
        job_id = self.start_job(peers[0], model_id, network_pipe.pipe_id, messages, tokens)
        self.jobs_pending.append(PendingJob(job_id, promise))
        return job_id
