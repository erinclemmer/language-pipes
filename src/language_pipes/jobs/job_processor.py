from enum import Enum, auto
from dataclasses import dataclass
from logging import Logger

from language_pipes.jobs.job import Job

from language_pipes.pipes.pipe import Pipe
from language_pipes.modeling.end_model import EndModel

from language_pipes.util.enums import ComputeStep, JobStatus
from language_pipes.config import LpConfig

class JobState(Enum):
    VALIDATING = auto()    # Validating pipe and getting resources
    HEAD = auto()          # Computing norm/head, handling completion
    EMBED = auto()         # Embedding the next token for decoding
    PROCESS_LAYERS = auto() # Processing through local layers
    SEND = auto()          # Sending job to next destination
    DONE = auto()          # Current job iteration complete

@dataclass
class JobContext:
    job: Job
    pipe: Pipe
    logger: Logger
    config: LpConfig
    end_model: EndModel

def should_prefill_chunk(job) -> bool:
    return job.current_token == 0 and job.chunking.has_more()

def log_done_error(ctx: JobContext, message: str) -> None:
    if ctx.logger is not None:
        ctx.logger.error(message)

def get_next_state(ctx: JobContext) -> JobState:
    cs = ctx.job.compute_step
    if cs == ComputeStep.HEAD or cs == ComputeStep.EMBED or cs == ComputeStep.TOKENIZE:
        if ctx.job.origin_node_id != ctx.config.node_id:
            return JobState.SEND
        
        if should_prefill_chunk(ctx.job) or cs == ComputeStep.EMBED or cs == ComputeStep.TOKENIZE:
            return JobState.EMBED  
        else:
            return JobState.HEAD

    model = ctx.pipe.get_layer(ctx.job.current_layer, False)
    if model is None:
        log_done_error(
            ctx,
            f"[FSM] Missing model for layer={ctx.job.current_layer}; completing with error."
        )
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
            log_done_error(self.ctx, "[FSM] Missing job; completing with error.")
            return JobState.DONE
        
        pipe = self.ctx.pipe
        
        # Ensure we have an available pipe
        if pipe is None or not pipe.is_complete():
            log_done_error(self.ctx, "[FSM] Pipe unavailable or incomplete; completing with error.")
            return JobState.DONE
        
        if self.ctx.job.compute_step == ComputeStep.HEAD:
            # Ensure we only process the ends of jobs we sent out
            if self.ctx.job.origin_node_id != self.ctx.config.node_id:
                log_done_error(self.ctx, "[FSM] Layer job origin mismatch; completing with error.")
                return JobState.DONE
            
            # Ensure we have the end model ready            
            if self.ctx.end_model is None:
                log_done_error(self.ctx, "[FSM] End model unavailable; completing with error.")
                return JobState.DONE
            
            # Job returned from network - check pending job
            if self.ctx.job is None:
                log_done_error(self.ctx, "[FSM] Job missing; completing with error.")
                return JobState.DONE

        return get_next_state(self.ctx)

    def _state_head(self) -> JobState:
        """Handle norm/head computation and prepare to embed the next token."""
        job = self.ctx.job
        end_model = self.ctx.end_model

        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if job.chunking.has_more():
                log_done_error(self.ctx, "Received head state for job that was not done chunking")
                return JobState.DONE

            job.chunking.disable()
        
        job.compute_step = ComputeStep.NORM
        job.current_layer = 0

        job.timing_stats.add_head_time(self.ctx.config.node_id)
        end_model.compute_norm(job)
        end_model.compute_head(job)
        job.timing_stats.set_send_time(self.ctx.logger)
        job.timing_stats.finalize_token(self.ctx.logger)
        job.print_job(self.ctx.logger)

        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            job.complete()
            return JobState.DONE
        
        # More tokens to generate - update and continue
        if not job.send_update():
            log_done_error(self.ctx, "[FSM] Failed to send job update; completing with error.")
            return JobState.DONE

        return JobState.EMBED

    def _state_embed(self) -> JobState:
        """Embed the next token, handling prefill chunks when needed."""
        job = self.ctx.job
        end_model = self.ctx.end_model
        logger = self.ctx.logger

        if job.prompt_tokens == 0:
            end_model.tokenize(job)
            logger.info(f"[Prefill] job={job.job_id[:8]} started")
            job.init_chunking(self.ctx.config.prefill_chunk_size)
            job.chunking.print_start(logger)
        elif job.chunking.is_active():
            job.chunking.advance()
            job.timing_stats.finalize_prefill_chunk(logger)
            logger.info(f"[PREFILL] Chunk {job.chunking.current_chunk} / {job.chunking.total_chunks} ({(job.chunking.current_chunk / job.chunking.total_chunks) * 100.0:.2f}%)")
            job.delta = ""
            if not job.send_update():
                log_done_error(self.ctx, "[FSM] Failed to send prefill job update; completing with error.")
                return JobState.DONE
        
        job.timing_stats.add_embed_time(self.ctx.config.node_id)
        end_model.compute_embed(job, logger, self.ctx.config.prefill_chunk_size)
        job.timing_stats.set_send_time(logger)
        
        return get_next_state(self.ctx)

    def _state_process_layers(self) -> JobState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        job = self.ctx.job
        
        model = pipe.get_layer(job.current_layer, True)
        if model is None:
            log_done_error(
                self.ctx,
                f"[FSM] Missing local model for layer={job.current_layer}; completing with error."
            )
            return JobState.DONE
        
        model.process_job(job, self.ctx.logger)
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
