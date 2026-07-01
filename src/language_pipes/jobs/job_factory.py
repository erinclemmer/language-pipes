from time import time

from promise import Promise
from typing import List, Optional, Callable

from language_pipes.jobs.job import Job
from language_pipes.util.chat import ChatMessage
from language_pipes.jobs.job_tracker import JobTracker
from language_pipes.pipes.pipe_manager import PipeManager
from language_pipes.util.config import get_max_api_jobs

class JobFactory:
    job_tracker: JobTracker
    pipe_manager: PipeManager

    def __init__(
        self, 
        job_tracker: JobTracker,
        pipe_manager: PipeManager
    ):
        self.job_tracker = job_tracker
        self.pipe_manager = pipe_manager
        self.logs = []

    def start_job(
        self, 
        api_key: str,
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
        end_model = self.pipe_manager.model_manager.get_end_model(model_id)
        if end_model is None:
            if resolve is not None:
                resolve('NO_ENDS') # pyright: ignore[reportCallIssue]
            return
        
        pipe = self.pipe_manager.get_pipe_by_model_id(model_id, start_layer=len(end_model.layers))
        if pipe is None:
            if resolve is not None:
                resolve('NO_PIPE') # pyright: ignore[reportCallIssue]
            return

        if api_key in self.job_tracker.jobs_pending and len(self.job_tracker.jobs_pending[api_key]) > get_max_api_jobs():
            if resolve is not None:
                resolve('MAX_JOBS') # pyright: ignore[reportCallIssue]
            return
        
        node_id = self.pipe_manager.router_pipes.router.node_id()

        job = Job(
            origin_node_id=node_id,
            messages=messages, 
            pipe_id=pipe.pipe_id, 
            model_id=pipe.model_id,
            config=end_model.collector.config,
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            resolve=resolve,
            update=update,
            complete=self.job_tracker.complete_job
        )

        self.logs.append((time(), f"Job {job.job_id[:4]} started"))
        
        network_job = job.to_network_job()
        pipe.send_job(network_job, node_id)
        if api_key not in self.job_tracker.jobs_pending:
            self.job_tracker.jobs_pending[api_key] = [ ]

        self.job_tracker.jobs_pending[api_key].append(job)

        if start is not None:
            start(job)

        return job

