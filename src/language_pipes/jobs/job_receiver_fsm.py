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
    PREFILL_CHUNK = auto() # Processing prefill chunks
    OUTPUT = auto()        # Computing norm/head, handling completion
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

def embed(
    ctx: FSMContext,
    chunk_start: int = 0,
    chunk_end: int = -1
) -> LayerJob:
    lt = create_embed_time(ctx.layer_job.origin_node_id)
    ctx.end_model.compute_embed(ctx.pending_job.job, ctx.pending_job.cache, chunk_start, chunk_end)
    lt.set_send_time()
    layer_job = ctx.pending_job.job.to_layer_job()
    layer_job.times.append(lt)
    return layer_job

def get_next_state(node_id: str, ctx: FSMContext) -> ReceiverState:
    if ctx.layer_job.done:
        if ctx.layer_job.origin_node_id == node_id:
            if ctx.pending_job.job.current_token == 0 and ctx.pending_job.chunking.has_more():
                return ReceiverState.PREFILL_CHUNK
            else:
                return ReceiverState.OUTPUT
        else:
            return ReceiverState.SEND
    
    model = ctx.pipe.get_layer(ctx.layer_job.current_layer, False)
    if model is None:
        return ReceiverState.DONE
    
    if model.virtual:
        return ReceiverState.SEND
    else:
        return ReceiverState.PROCESS_LAYERS

class JobReceiverFSM:
    """
    Finite state machine for processing jobs.
    
    State Transitions:
    
    VALIDATING -> DONE (Error occurred)
    VALIDATING -> OUTPUT (job.done and final chunk or decode)
    VALIDATING -> PREFILL_CHUNK (job.done and more chunks)
    VALIDATING -> PROCESS_LAYERS (layer job not done)
    
    PREFILL_CHUNK -> DONE (Error occurred)
    PREFILL_CHUNK -> SEND (first layer is virtual)
    PREFILL_CHUNK -> PROCESS_LAYERS (first layer is local)
    
    OUTPUT -> DONE (Error occurred)
    OUTPUT -> SEND (first layer is virtual)
    OUTPUT -> PROCESS_LAYERS (first layer is local)
    
    PROCESS_LAYERS -> DONE (Error occurred)
    PROCESS_LAYERS -> SEND (next layer set is not local)
    PROCESS_LAYERS -> PROCESS_LAYERS (next layer set is local)
    
    SEND -> DONE (always)
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
            case ReceiverState.PREFILL_CHUNK:
                return self._state_prefill_chunk()
            case ReceiverState.OUTPUT:
                return self._state_output()
            case ReceiverState.PROCESS_LAYERS:
                return self._state_process_layers()
            case ReceiverState.SEND:
                return self._state_send()
    
    def _state_validating(self) -> ReceiverState:
        """Validate pipe availability and gather resources."""
        layer_job = self.ctx.layer_job
        if layer_job is None:
            self.state = ReceiverState.DONE
            return
        
        pipe = self.ctx.pipe
        
        # Pipe unavailable - drop job
        if pipe is None or not pipe.is_complete():
            return ReceiverState.DONE
        
        if layer_job.done:
            # Ensure we only process the ends of jobs we sent out
            if layer_job.origin_node_id != self.node_id:
                return ReceiverState.DONE
            
            # Ensure we have the end model ready            
            if self.ctx.end_model is None:
                return ReceiverState.DONE
            
            # Job returned from network - check pending job
            if self.ctx.pending_job is None:
                return ReceiverState.DONE
            
            job = self.ctx.pending_job.job
            job.data = layer_job.data

            # Prefill chunking: more chunks?
            if job.current_token == 0 and self.ctx.pending_job.chunking.has_more():
                return ReceiverState.PREFILL_CHUNK
            else:
                return ReceiverState.OUTPUT
        else:
            job = self.ctx.pending_job.job
            if job.current_token == 0 and self.ctx.pending_job.chunking.has_more():
                return ReceiverState.PREFILL_CHUNK
            else:
                return ReceiverState.PROCESS_LAYERS

    def _state_prefill_chunk(self) -> ReceiverState:
        """Handle the next prefill chunk."""
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        job = pending_job.job
        
        # Update job time to prevent stale timeout during prefill
        pending_job.set_last_update()
        
        # Log chunk completion
        chunk_time_ms = (time() - job.chunk_start_time) * 1000
        self.ctx.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
        )
        
        chunk_start, chunk_end = pending_job.chunking.get_range()
        pending_job.chunking.advance()
        
        # Log next chunk start
        job.chunk_start_time = time()
        self.ctx.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
        )
        
        job.current_step = ComputeStep.EMBED
        self.ctx.layer_job = embed(
            self.ctx, chunk_start, chunk_end
        )
        self.ctx.layer_job.done = False

        job.delta = ""
        if not pipe.send_job_update(job):
            return ReceiverState.DONE
        
        return get_next_state(self.node_id, self.ctx)
    
    def _state_output(self) -> ReceiverState:
        """Handle norm/head computation and prepare next token."""
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        job = pending_job.job
        
        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if pending_job.chunking.is_active():
                chunk_time_ms = (time() - job.chunk_start_time) * 1000
                self.ctx.logger.info(
                    f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
                    f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
                )
            
            total_prefill_ms = (time() - job.prefill_start_time) * 1000
            tokens_per_sec = (job.prompt_tokens / total_prefill_ms) * 1000 if total_prefill_ms > 0 else 0
            self.ctx.logger.info(
                f"[Prefill] job={job.job_id[:8]} finished: "
                f"prompt_tokens={job.prompt_tokens}, "
                f"total_time={total_prefill_ms:.1f}ms, "
                f"throughput={tokens_per_sec:.1f} tok/s"
            )
        
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
            return ReceiverState.DONE
        
        # Embed next token (decode phase - no chunking)
        self.ctx.layer_job = embed(self.ctx)

        return get_next_state(self.node_id, self.ctx)

    def _state_process_layers(self) -> ReceiverState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        
        model = pipe.get_layer(layer_job.current_layer, True)
        if model is None:
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