from time import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from language_pipes.jobs.layer_job import LayerJob, LayerTime

from language_pipes.pipes.pipe import Pipe
from language_pipes.modeling.end_model import EndModel

from language_pipes.util.enums import ComputeStep, JobStatus

class ReceiverState(Enum):
    """States for the JobReceiver finite state machine."""
    VALIDATING = auto()    # Validating pipe and getting resources
    HEAD = auto()          # Computing norm/head, handling completion
    EMBED = auto()         # Embedding the next token for decoding
    PROCESS_LAYERS = auto() # Processing through local layers
    SEND = auto()          # Sending job to next destination
    DONE = auto()          # Current job iteration complete

@dataclass
class FSMContext:
    """Context passed between FSM states."""
    logger: Optional[any] = None
    job: Optional["Job"] = None
    pipe: Optional[Pipe] = None
    end_model: Optional[EndModel] = None

def create_embed_time(node_id: str) -> LayerTime:
    return LayerTime(node_id=node_id, is_embed=True)

def create_head_time(node_id: str) -> LayerTime:
    """Create a LayerTime for head operations."""
    return LayerTime(node_id=node_id, is_head=True)

def embed(ctx: FSMContext):
    if ctx.job.chunking.is_active() and ctx.job.current_token == 0:
        chunk_start, chunk_end = ctx.job.chunking.get_range()
    else:
        chunk_start, chunk_end = (0, -1)
    ctx.end_model.compute_embed(ctx.job, ctx.job.cache, chunk_start, chunk_end)

def should_prefill_chunk(job) -> bool:
    return job.current_token == 0 and job.chunking.has_more()

def log_prefill_chunk_complete(logger, job) -> None:
    chunk_time_ms = (time() - job.chunk_start_time) * 1000
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {job.chunking.current_chunk + 1}/"
        f"{job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
    )

def log_prefill_chunk_start(logger, job, chunk_start: int, chunk_end: int) -> None:
    job.chunk_start_time = time()
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {job.chunking.current_chunk + 1}/"
        f"{job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
    )

def log_prefill_summary(logger, job) -> None:
    total_prefill_ms = (time() - job.prefill_start_time) * 1000
    tokens_per_sec = (job.prompt_tokens / total_prefill_ms) * 1000 if total_prefill_ms > 0 else 0
    logger.info(
        f"[Prefill] job={job.job_id[:8]} finished: "
        f"prompt_tokens={job.prompt_tokens}, "
        f"total_time={total_prefill_ms:.1f}ms, "
        f"throughput={tokens_per_sec:.1f} tok/s"
    )

def log_done_error(ctx: FSMContext, message: str) -> None:
    if ctx.logger is not None:
        ctx.logger.error(message)

def get_next_state(node_id: str, ctx: FSMContext) -> ReceiverState:
    if ctx.job.compute_step == ComputeStep.HEAD or ctx.job.compute_step == ComputeStep.EMBED:
        if ctx.job.origin_node_id != node_id:
            return ReceiverState.SEND
        return ReceiverState.EMBED if should_prefill_chunk(ctx.job) or ctx.job.compute_step == ComputeStep.EMBED else ReceiverState.HEAD

    model = ctx.pipe.get_layer(ctx.job.current_layer, False)
    if model is None:
        log_done_error(
            ctx,
            f"[FSM] Missing model for layer={ctx.job.current_layer}; completing with error."
        )
        return ReceiverState.DONE

    if model.virtual:
        return ReceiverState.SEND
    return ReceiverState.PROCESS_LAYERS

class JobReceiverFSM:
    """
    Finite state machine for processing jobs.
    
    State Transitions (and why they happen):
    
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
    
    state: ReceiverState
    ctx: FSMContext
    receiver: 'JobReceiver'
    
    def __init__(
        self,
        node_id: str,
        print_job: bool,
        print_times: bool,
    ):
        self.node_id = node_id
        self.print_times = print_times
        self.print_job = print_job
        self.state = ReceiverState.VALIDATING
        self.ctx = FSMContext()
    
    def run(self) -> bool:
        while self.state != ReceiverState.DONE:
            self.state = self._transition()
    
    def _transition(self) -> ReceiverState:
        """Execute current state and transition to next."""
        match self.state:
            case ReceiverState.VALIDATING:
                return self._state_validating()
            case ReceiverState.HEAD:
                return self._state_head()
            case ReceiverState.EMBED:
                return self._state_embed()
            case ReceiverState.PROCESS_LAYERS:
                return self._state_process_layers()
            case ReceiverState.SEND:
                return self._state_send()
    
    def _state_validating(self) -> ReceiverState:
        """Validate context for processing"""
        if self.ctx.job is None:
            log_done_error(self.ctx, "[FSM] Missing job; completing with error.")
            return ReceiverState.DONE
        
        pipe = self.ctx.pipe
        
        # Ensure we have an available pipe
        if pipe is None or not pipe.is_complete():
            log_done_error(self.ctx, "[FSM] Pipe unavailable or incomplete; completing with error.")
            return ReceiverState.DONE
        
        if self.ctx.job.compute_step == ComputeStep.HEAD:
            # Ensure we only process the ends of jobs we sent out
            if self.ctx.job.origin_node_id != self.node_id:
                log_done_error(self.ctx, "[FSM] Layer job origin mismatch; completing with error.")
                return ReceiverState.DONE
            
            # Ensure we have the end model ready            
            if self.ctx.end_model is None:
                log_done_error(self.ctx, "[FSM] End model unavailable; completing with error.")
                return ReceiverState.DONE
            
            # Job returned from network - check pending job
            if self.ctx.job is None:
                log_done_error(self.ctx, "[FSM] Job missing; completing with error.")
                return ReceiverState.DONE

        return get_next_state(self.node_id, self.ctx)

    def _state_head(self) -> ReceiverState:
        """Handle norm/head computation and prepare to embed the next token."""
        job = self.ctx.job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model

        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if job.chunking.has_more():
                log_done_error(self.ctx, "Received head state for job that was not done chunking")
                return ReceiverState.DONE

            if job.chunking.is_active():
                log_prefill_chunk_complete(self.ctx.logger, job)

            log_prefill_summary(self.ctx.logger, job)
            job.chunking.disable()
        
        job.compute_step = ComputeStep.NORM
        job.current_layer = 0
        
        end_model.compute_norm(job)
        end_model.compute_head(job)
        
        if self.print_job:
            job.print_job(self.ctx.logger)

        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            pipe.complete_job(job)
            return ReceiverState.DONE
        
        # More tokens to generate - update and continue
        if not pipe.send_job_update(job):
            log_done_error(self.ctx, "[FSM] Failed to send job update; completing with error.")
            return ReceiverState.DONE

        return ReceiverState.EMBED

    def _state_embed(self) -> ReceiverState:
        """Embed the next token, handling prefill chunks when needed."""
        job = self.ctx.job
        if should_prefill_chunk(job):
            job.set_last_update()
            log_prefill_chunk_complete(self.ctx.logger, job)
            job.chunking.advance()
            chunk_start, chunk_end = job.chunking.get_range()
            log_prefill_chunk_start(self.ctx.logger, job, chunk_start, chunk_end)

        embed(self.ctx)
        if should_prefill_chunk(job) or job.chunking.is_active():
            job.delta = ""
            if not self.ctx.pipe.send_job_update(job):
                log_done_error(self.ctx, "[FSM] Failed to send prefill job update; completing with error.")
                return ReceiverState.DONE
        return get_next_state(self.node_id, self.ctx)

    def _state_process_layers(self) -> ReceiverState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        job = self.ctx.job
        
        model = pipe.get_layer(job.current_layer, True)
        if model is None:
            log_done_error(
                self.ctx,
                f"[FSM] Missing local model for layer={job.current_layer}; completing with error."
            )
            return ReceiverState.DONE
        
        model.process_job(job, job.cache)
        job.set_last_update()
        
        return get_next_state(self.node_id, self.ctx)
    
    def _state_send(self) -> ReceiverState:
        """Send job to next destination."""
        job = self.ctx.job
        pipe = self.ctx.pipe
        layer_job = job.to_layer_job()

        if job.compute_step == ComputeStep.HEAD:
            pipe.send_job(layer_job, layer_job.origin_node_id)
        else:
            next_model = pipe.get_layer(layer_job.current_layer, False)
            pipe.send_job(layer_job, next_model.node_id)
        
        return ReceiverState.DONE
