from time import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from language_pipes.jobs.layer_job import LayerJob, LayerTime
from language_pipes.jobs.pending_job import PendingJob

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
    layer_job: Optional[LayerJob] = None
    pending_job: Optional[PendingJob] = None
    pipe: Optional[Pipe] = None
    end_model: Optional[EndModel] = None

def create_embed_time(node_id: str) -> LayerTime:
    return LayerTime(node_id=node_id, is_embed=True)

def create_head_time(node_id: str) -> LayerTime:
    """Create a LayerTime for head operations."""
    return LayerTime(node_id=node_id, is_head=True)

def embed(ctx: FSMContext) -> LayerJob:
    lt = create_embed_time(ctx.layer_job.origin_node_id)
    if ctx.pending_job.chunking.is_active() and ctx.pending_job.job.current_token == 0:
        chunk_start, chunk_end = ctx.pending_job.chunking.get_range()
    else:
        chunk_start, chunk_end = (0, -1)
    ctx.end_model.compute_embed(ctx.pending_job.job, ctx.pending_job.cache, chunk_start, chunk_end)
    lt.set_send_time()
    layer_job = ctx.pending_job.job.to_layer_job()
    layer_job.times.append(lt)
    return layer_job

def should_prefill_chunk(pending_job: PendingJob) -> bool:
    return pending_job.job.current_token == 0 and pending_job.chunking.has_more()

def log_prefill_chunk_complete(logger, job, pending_job) -> None:
    chunk_time_ms = (time() - job.chunk_start_time) * 1000
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
        f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
    )

def log_prefill_chunk_start(logger, job, pending_job, chunk_start: int, chunk_end: int) -> None:
    job.chunk_start_time = time()
    logger.info(
        f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
        f"{pending_job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
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

def disable_chunking(pending_job: PendingJob) -> None:
    pending_job.chunking.current_chunk = 0
    pending_job.chunking.total_chunks = 0
    pending_job.chunking.chunk_size = 0

def log_done_error(ctx: FSMContext, message: str) -> None:
    if ctx.logger is not None:
        ctx.logger.error(message)

def next_state_after_origin_completion(ctx: FSMContext) -> ReceiverState:
    return ReceiverState.EMBED if should_prefill_chunk(ctx.pending_job) else ReceiverState.HEAD

def next_state_after_local_job(ctx: FSMContext) -> ReceiverState:
    return ReceiverState.EMBED if should_prefill_chunk(ctx.pending_job) else ReceiverState.PROCESS_LAYERS

def get_next_state(node_id: str, ctx: FSMContext) -> ReceiverState:
    if ctx.layer_job.done:
        if ctx.layer_job.origin_node_id != node_id:
            return ReceiverState.SEND
        return next_state_after_origin_completion(ctx)
    
    model = ctx.pipe.get_layer(ctx.layer_job.current_layer, False)
    if model is None:
        log_done_error(
            ctx,
            f"[FSM] Missing model for layer={ctx.layer_job.current_layer}; completing with error."
        )
        return ReceiverState.DONE
    
    if model.virtual:
        return ReceiverState.SEND
    else:
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
        """Validate pipe availability and gather resources."""
        layer_job = self.ctx.layer_job
        if layer_job is None:
            log_done_error(self.ctx, "[FSM] Missing layer job; completing with error.")
            return ReceiverState.DONE
        
        pipe = self.ctx.pipe
        
        # Pipe unavailable - drop job
        if pipe is None or not pipe.is_complete():
            log_done_error(self.ctx, "[FSM] Pipe unavailable or incomplete; completing with error.")
            return ReceiverState.DONE
        
        if layer_job.done:
            # Ensure we only process the ends of jobs we sent out
            if layer_job.origin_node_id != self.node_id:
                log_done_error(self.ctx, "[FSM] Layer job origin mismatch; completing with error.")
                return ReceiverState.DONE
            
            # Ensure we have the end model ready            
            if self.ctx.end_model is None:
                log_done_error(self.ctx, "[FSM] End model unavailable; completing with error.")
                return ReceiverState.DONE
            
            # Job returned from network - check pending job
            if self.ctx.pending_job is None:
                log_done_error(self.ctx, "[FSM] Pending job missing; completing with error.")
                return ReceiverState.DONE
            
            self.ctx.pending_job.job.data = layer_job.data

            # Prefill chunking: more chunks?
            return next_state_after_origin_completion(self.ctx)
        return next_state_after_local_job(self.ctx)

    def _state_head(self) -> ReceiverState:
        """Handle norm/head computation and prepare to embed the next token."""
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        job = pending_job.job
        
        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if pending_job.chunking.is_active():
                log_prefill_chunk_complete(self.ctx.logger, job, pending_job)
            
            log_prefill_summary(self.ctx.logger, job)
            disable_chunking(pending_job)
        
        job.current_step = ComputeStep.NORM
        
        lt = create_head_time(self.node_id)
        end_model.compute_norm(job)
        end_model.compute_head(job)
        lt.set_send_time()
        layer_job.times.append(lt)
        
        if self.print_times:
            layer_job.print_times(self.ctx.logger)

        if self.print_job:
            job.print_job(self.ctx.logger)

        layer_job.times = []
        
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
        pending_job = self.ctx.pending_job
        job = pending_job.job
        if should_prefill_chunk(pending_job):
            pending_job.set_last_update()
            log_prefill_chunk_complete(self.ctx.logger, job, pending_job)
            pending_job.chunking.advance()
            chunk_start, chunk_end = pending_job.chunking.get_range()
            log_prefill_chunk_start(self.ctx.logger, job, pending_job, chunk_start, chunk_end)
        job.current_step = ComputeStep.EMBED
        self.ctx.layer_job = embed(self.ctx)
        self.ctx.layer_job.done = False
        if should_prefill_chunk(pending_job) or pending_job.chunking.is_active():
            job.delta = ""
            if not self.ctx.pipe.send_job_update(job):
                log_done_error(self.ctx, "[FSM] Failed to send prefill job update; completing with error.")
                return ReceiverState.DONE
        return get_next_state(self.node_id, self.ctx)

    def _state_process_layers(self) -> ReceiverState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        
        model = pipe.get_layer(layer_job.current_layer, True)
        if model is None:
            log_done_error(
                self.ctx,
                f"[FSM] Missing local model for layer={layer_job.current_layer}; completing with error."
            )
            return ReceiverState.DONE
        
        model.process_job(layer_job, pending_job.cache)
        
        # Only update pending job time for layer-only nodes (not the origin node)
        if layer_job.origin_node_id != self.node_id:
            pending_job.set_last_update()
        
        return get_next_state(self.node_id, self.ctx)
    
    def _state_send(self) -> ReceiverState:
        """Send job to next destination."""
        layer_job = self.ctx.layer_job
        pipe = self.ctx.pipe

        if layer_job.done:
            pipe.send_job(layer_job, layer_job.origin_node_id)
        else:
            next_model = pipe.get_layer(layer_job.current_layer, False)
            pipe.send_job(layer_job, next_model.node_id)
        
        return ReceiverState.DONE
