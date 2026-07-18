import logging
from typing import Optional
from enum import Enum, auto
from dataclasses import dataclass

from language_pipes.jobs.job import Job
from language_pipes.pipes.pipe import Pipe
from language_pipes.modeling.end_model import EndModel
from language_pipes.util.enums import ComputeStep, JobStatus

class JobState(Enum):
    VALIDATING = auto()    # Validating pipe and getting resources
    HEAD = auto()          # Computing norm/head, handling completion
    EMBED = auto()         # Embedding the next token for decoding
    PROCESS_LAYERS = auto() # Processing through local layers
    SEND = auto()          # Sending job to next destination
    DONE = auto()          # Current job iteration complete

@dataclass
class JobContext:
    node_id: str
    job: Job
    pipe: Pipe
    end_model: Optional[EndModel]

def should_prefill_chunk(job: Job) -> bool:
    return job.current_token == 0 and job.chunking.has_more()

def get_next_state(ctx: JobContext) -> JobState:
    cs = ctx.job.compute_step
    if cs == ComputeStep.HEAD or cs == ComputeStep.EMBED or cs == ComputeStep.TOKENIZE:
        if ctx.job.origin_node_id != ctx.node_id:
            return JobState.SEND
        
        if should_prefill_chunk(ctx.job) or cs == ComputeStep.EMBED or cs == ComputeStep.TOKENIZE:
            return JobState.EMBED  
        else:
            return JobState.HEAD

    if ctx.job.current_layer == 0 and ctx.end_model is not None and len(ctx.end_model.layers) > 0:
        return JobState.PROCESS_LAYERS

    model = ctx.pipe.get_layer(ctx.job.current_layer, False)
    if model is None:
        return JobState.DONE

    if model.virtual:
        return JobState.SEND
    return JobState.PROCESS_LAYERS

class JobProcessor:
    """
    Finite state machine for processing jobs.
    
    State Transitions:
    
    VALIDATING -> DONE (missing job/context resources or pipe unavailable)
    VALIDATING -> HEAD (job.done and prefill finished or decode)
    VALIDATING -> EMBED (job.done and more prefill chunks)
    VALIDATING -> PROCESS_LAYERS (job still needs local layer processing)
    
    HEAD -> DONE (job complete or failed to send update)
    HEAD -> EMBED (more tokens to generate locally)
    HEAD -> SEND (next layer is virtual/remote)
    HEAD -> PROCESS_LAYERS (next layer is local)

    EMBED -> DONE (failed to send update or missing model)
    EMBED -> SEND (next layer is virtual/remote)
    EMBED -> PROCESS_LAYERS (next layer is local)
    
    PROCESS_LAYERS -> DONE (missing local model)
    PROCESS_LAYERS -> SEND (next layer set is not local)
    PROCESS_LAYERS -> PROCESS_LAYERS (next layer set is local)
    
    SEND -> DONE (handoff complete)
    """
    
    state: JobState
    ctx: JobContext
    
    def __init__(self, ctx: JobContext):
        self.state = JobState.VALIDATING
        self.ctx = ctx
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        while self.state != JobState.DONE:
            self.state = self._transition()
    
    def _transition(self) -> JobState:
        """Execute current state and transition to next."""
        match self.state:
            case JobState.VALIDATING:
                return self._state_validating()
            case JobState.HEAD:
                return self._state_head()
            case JobState.EMBED:
                return self._state_embed()
            case JobState.PROCESS_LAYERS:
                return self._state_process_layers()
            case JobState.SEND:
                return self._state_send()

        return JobState.DONE
    
    def _state_validating(self) -> JobState:
        """Validate context for processing"""
        if self.ctx.job is None:
            return JobState.DONE
        
        if self.ctx.job.compute_step == ComputeStep.HEAD:
            # Ensure we only process the ends of jobs we sent out
            if self.ctx.job.origin_node_id != self.ctx.node_id:
                return JobState.DONE
            
            # Ensure we have the end model ready            
            if self.ctx.end_model is None:
                return JobState.DONE
            
            # Job returned from network - check pending job
            if self.ctx.job is None:
                return JobState.DONE

        return get_next_state(self.ctx)

    def _state_head(self) -> JobState:
        """Handle norm/head computation and prepare to embed the next token."""
        job = self.ctx.job
        end_model = self.ctx.end_model
        if end_model is None:
            return JobState.DONE

        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if job.chunking.has_more():
                return JobState.DONE

            job.chunking.disable()
        
        job.compute_step = ComputeStep.NORM
        job.current_layer = 0

        job.timing_stats.add_head_time(self.ctx.node_id)
        end_model.compute_norm(job)
        end_model.compute_head(job)
        job.timing_stats.set_send_time()
        job.timing_stats.finalize_token()

        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            job.complete()
            self.logger.info(f"Job {job.job_id[:4]} completed")
            return JobState.DONE
        
        # More tokens to generate - update and continue
        if not job.send_update():
            job.stale = True
            job.status = JobStatus.COMPLETED
            end_model.set_result(job)
            job.complete()
            self.logger.info(f"Job {job.job_id[:4]} completed")
            return JobState.DONE

        return JobState.EMBED

    def _state_embed(self) -> JobState:
        """Embed the next token, handling prefill chunks when needed."""
        job = self.ctx.job
        end_model = self.ctx.end_model

        if end_model is None:
            return JobState.DONE

        if job.prompt_tokens == 0:
            end_model.tokenize(job)
            job.init_chunking()
        elif job.chunking.is_active():
            job.chunking.advance()
            job.timing_stats.finalize_prefill_chunk()
            job.delta = ""
            if not job.send_update():
                job.stale = True
                job.status = JobStatus.COMPLETED
                end_model.set_result(job)
                job.complete()
                return JobState.DONE
        
        job.set_last_update()
        job.timing_stats.add_embed_time(self.ctx.node_id)
        end_model.compute_embed(job)
        job.timing_stats.set_send_time()
        
        return get_next_state(self.ctx)

    def _state_process_layers(self) -> JobState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        job = self.ctx.job

        if job.current_layer == 0 and self.ctx.end_model is not None and len(self.ctx.end_model.layers) > 0:
            self.ctx.end_model.compute_layers(job)

        model = pipe.get_layer(job.current_layer, False)
        if model is None:
            return JobState.DONE
        
        if model.virtual:
            return JobState.SEND
        
        model.process_job(job)
        job.set_last_update()
        
        return get_next_state(self.ctx)
    
    def _state_send(self) -> JobState:
        """Send job to next destination."""
        job = self.ctx.job
        pipe = self.ctx.pipe
        network_job = job.to_network_job()

        if job.compute_step == ComputeStep.HEAD:
            pipe.send_job(network_job, network_job.origin_node_id)
        else:
            next_model = pipe.get_layer(network_job.current_layer, False)
            if next_model is None:
                return JobState.DONE
            pipe.send_job(network_job, next_model.node_id)
        
        return JobState.DONE
