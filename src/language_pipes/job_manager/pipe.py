import os
import requests
from threading import Thread
from typing import Callable, List, Optional
from uuid import uuid4

from transformers import AutoTokenizer
from transformers.models.auto import AutoConfig
from distributed_state_network import DSNode

from language_pipes.util.meta import MetaPipe
from language_pipes.util.chat import ChatMessage
from language_pipes.job_manager.enums import JobStatus
from language_pipes.llm_model import LlmModel
from language_pipes.job_manager.job import Job, ComputeStep

class Pipe:
    pipe_id: str
    model_id: str
    segments: List[LlmModel]

    router: DSNode
    tokenizer: Callable
    get_job_port: Callable[[str], Optional[int]]
    complete_job: Callable[[Job], None]
    model_num_hidden_layers: int

    def __init__(
            self, 
            router: DSNode,
            pipe_id: Optional[str],
            model_id: str,
            https: bool,
            get_job_port: Callable[[str], Optional[int]],
            complete_job: Callable[[Job], None],
            restart_job: Callable[[Job], None]
        ):
        self.get_job_port = get_job_port
        self.complete_job = complete_job
        self.restart_job = restart_job
        self.router = router
        self.https = https
        self.model_id = model_id
        self.model_num_hidden_layers = AutoConfig.from_pretrained(f'./models/{model_id}/data').num_hidden_layers
        
        if pipe_id is None:
            self.pipe_id = str(uuid4())
        else:
            self.pipe_id = pipe_id

        self.segments = []
        self.tokenizer = lambda: AutoTokenizer.from_pretrained(os.path.join('models', model_id, 'data'))

    def raise_exception(self, msg: str):
        self.router.logger.exception(msg)
        raise Exception(msg)

    def send_job(self, job: Job, router_id: str):
        try:
            ip = self.router.connection_from_node(router_id).address
            port = self.get_job_port(router_id)
            if port is None:
                self.raise_exception(f"SEND JOB => Could not find pipe {self.pipe_id} for {router_id}")

            job.from_router_id = self.router.config.node_id
            job.sign(self.router.cred_manager.my_private())

            self.router.logger.info(f'Sending job {job.job_id} to {router_id} (token {job.current_token})')
            cert = self.router.cert_manager.public_path(router_id)
            if cert is None:
                self.raise_exception(f"SEND JOB => Could not find certificate for {router_id}")
            protocol = 'https' if self.https else 'http'
            if protocol == 'http':
                cert = None
            def send(url: str, data: bytes, cert: str):
                try:
                    res = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'}, verify=cert)
                    if res.status_code != 200 or res.content == b'DOWN':
                        self.raise_exception(f"SEND JOB => bad response from {router_id}")
                except:
                    self.raise_exception(f"SEND JOB => Could not connect to {router_id}")
            Thread(target=send, args=(f'{protocol}://{ip}:{port}', job.to_bytes(), cert, )).start()
        except:
            self.restart_job(job)

    def tokenize(self, prompt: Optional[str], messages: List[ChatMessage]) -> List[int]:
        tokenizer: AutoTokenizer = self.tokenizer()
        if prompt is None:
            prompt = tokenizer.apply_chat_template([m.to_json() for m in messages], tokenize=False, add_generation_prompt=True, chat_template=tokenizer.chat_template)
        return [int(t) for t in tokenizer.encode(prompt, return_tensors='pt')[0].numpy()]

    def get_embed(self, need_physical: bool = False) -> Optional[LlmModel]:
        res = list(filter(lambda m: m.loaded and m.input_embedding is not None and (not need_physical or not m.virtual), self.segments))
        if len(res) == 0:
            return None
        return res[0]
    
    def get_head(self, need_physical: bool = False) -> Optional[LlmModel]:
        res = list(filter(lambda m: m.loaded and m.head is not None and (not need_physical or not m.virtual), self.segments))
        if len(res) == 0:
            return None
        return res[0]

    def get_layer(self, layer: int, need_physical: bool = False) -> Optional[LlmModel]:
        for segment in self.segments:
            if segment.start_layer == layer and (not need_physical or not segment.virtual):
                return segment
        return None
    
    def get_computed(self):
        return self.segments[0].computed

    def sort_segments(self):
        self.segments = sorted(self.segments, key=lambda x: x.start_layer)

    def is_complete(self):
        self.sort_segments()
        if self.get_embed() is None or self.get_head() is None:
            return False
        current_layer = 0
        for s in self.segments:
            if s.start_layer == current_layer:
                current_layer = s.end_layer + 1

        return current_layer == self.model_num_hidden_layers

    def print(self):
        self.router.logger.info(f'''
#################################
Pipe Status:
Model ID: {self.model_id}
Pipe: {self.pipe_id}
Segments: {', '.join([s.router_id for s in self.segments])}
Embed: {not self.get_embed() is not None}
Head: {not self.get_head() is not None}
End Layer: {self.segments[-1].end_layer}
Complete: {self.is_complete()}
#################################
''')

    def peers(self) -> List[str]:
        peers: List[str] = []
        for segment in self.segments:
            if segment.router_id not in peers:
                peers.append(segment.router_id)
        return peers

    def model_for_job(self, job: Job, need_physical: bool = False) -> Optional[LlmModel]:
        model = None
        if job.status == JobStatus.COMPLETED:
            return None

        if job.current_step == ComputeStep.TOKENIZE:
            model = self.segments[0]
        if job.current_step == ComputeStep.EMBED:
            model = self.get_embed(need_physical)
        if job.current_step == ComputeStep.LAYER:
            model = self.get_layer(job.current_layer, need_physical)
        if job.current_step == ComputeStep.NORM:
            model = self.segments[0]
        if job.current_step == ComputeStep.HEAD:
            model = self.get_head(need_physical)
        return model

    def process_job(
            self,
            job: Job
        ):
        while True:
            model_for_job = self.model_for_job(job)
            if model_for_job is None:
                if job.status == JobStatus.COMPLETED:
                    if job.router_id == self.router.config.node_id:
                        job.result = self.tokenizer().decode(job.input_id_tensor())
                        self.complete_job(job)
                    else:
                        self.send_job(job, job.router_id)
                else:
                    self.restart_job(job)
                return
            
            if model_for_job.virtual:
                self.send_job(job, model_for_job.router_id)
                return
            
            model_for_job.process_job(job)

    @staticmethod
    def from_meta(
        meta_pipe: MetaPipe, 
        hosted_models: List[LlmModel], 
        router: DSNode,
        https: bool,
        get_job_port: Callable[[str], Optional[int]],
        complete_job: Callable[[Job], None],
        restart_job: Callable[[Job], None]
    ) -> 'Pipe':
        p = Pipe(
            model_id=meta_pipe.model_id, 
            pipe_id=meta_pipe.pipe_id, 
            https=https,
            get_job_port=get_job_port,
            complete_job=complete_job,
            restart_job=restart_job,
            router=router
        )
        local_segments = []
        for model in hosted_models:
            if model.pipe_id == meta_pipe.pipe_id:
                p.segments.append(model)
                local_segments.append(model.process_id)
        p.segments.extend([LlmModel.from_meta(s) for s in meta_pipe.segments if s.process_id not in local_segments])
        p.sort_segments()
        return p
