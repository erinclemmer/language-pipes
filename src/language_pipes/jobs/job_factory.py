from typing import List, Optional, Callable

from promise import Promise

from language_pipes.util import raise_exception
from language_pipes.util.chat import ChatMessage

from language_pipes.pipes.pipe_manager import PipeManager

from language_pipes.jobs.job import Job
from language_pipes.jobs.job_tracker import JobTracker

from language_pipes.config import LpConfig

from language_pipes.modeling.llm_meta_data import validate_model

class JobFactory:
    config: LpConfig
    job_tracker: JobTracker
    pipe_manager: PipeManager

    def __init__(
        self, 
        logger,
        config: LpConfig,
        job_tracker: JobTracker,
        pipe_manager: PipeManager
    ):
        self.config = config
        self.logger = logger
        self.job_tracker = job_tracker
        self.pipe_manager = pipe_manager

    def start_job(
        self, 
        model_id: str, 
        messages: List[ChatMessage], 
        max_completion_tokens: int, 
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        start: Optional[Callable] = None,
        update: Optional[Callable] = None,
        resolve: Optional[Promise] = None
    ) -> Optional[Job]:
        pipe = self.pipe_manager.get_pipe_by_model_id(model_id)
        if pipe is None:
            resolve('No pipe available')
            raise_exception(self.logger, f"Could not find pipe for model {model_id}")
            return

        job = Job(
            origin_node_id=self.config.node_id,
            messages=messages, 
            pipe_id=pipe.pipe_id, 
            model_id=pipe.model_id,
            temperature=temperature, 
            prefill_chunk_size=self.config.prefill_chunk_size,
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            resolve=resolve,
            update=update,
            complete=self.job_tracker.complete_job
        )
        
        job.print_job(self.logger)
        
        network_job = job.to_network_job()
        pipe.send_job(network_job, self.config.node_id)
        self.job_tracker.jobs_pending.append(job)

        if start is not None:
            start(job)

        return job
