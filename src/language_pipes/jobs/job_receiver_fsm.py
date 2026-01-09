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
    WAITING = auto()       # Waiting for a job from the queue
    VALIDATING = auto()    # Validating pipe and getting resources
    RESTART = auto()       # Handling a restart request
    PREFILL_CHUNK = auto() # Processing prefill chunks
    OUTPUT = auto()        # Computing norm/head, handling completion
    PROCESS_LAYERS = auto() # Processing through local layers
    SEND = auto()          # Sending job to next destination
    DONE = auto()          # Current job iteration complete


@dataclass
class FSMContext:
    """Context passed between FSM states."""
    layer_job: Optional[LayerJob] = None
    pending_job: Optional[PendingJob] = None
    pipe: Optional[Pipe] = None
    end_model: Optional[EndModel] = None


class JobReceiverFSM:
    """
    Finite state machine for processing jobs.
    
    State Transitions:
    
    WAITING -> VALIDATING (job received)
    WAITING -> DONE (shutting down)
    
    VALIDATING -> RESTART (restart flag set)
    VALIDATING -> OUTPUT (job.done and final chunk or decode)
    VALIDATING -> PREFILL_CHUNK (job.done and more chunks)
    VALIDATING -> PROCESS_LAYERS (job not done)
    VALIDATING -> DONE (pipe unavailable, restart queued)
    
    RESTART -> SEND (first layer is virtual)
    RESTART -> PROCESS_LAYERS (first layer is local)
    RESTART -> DONE (pending job not found)
    
    PREFILL_CHUNK -> SEND (first layer is virtual)
    PREFILL_CHUNK -> PROCESS_LAYERS (first layer is local)
    PREFILL_CHUNK -> DONE (model not found)
    
    OUTPUT -> DONE (job completed or update failed)
    OUTPUT -> SEND (first layer is virtual)
    OUTPUT -> PROCESS_LAYERS (first layer is local)
    
    PROCESS_LAYERS -> SEND (always)
    
    SEND -> DONE (always)
    """
    
    state: ReceiverState
    ctx: FSMContext
    receiver: 'JobReceiver'
    
    def __init__(self, receiver: 'JobReceiver'):
        self.receiver = receiver
        self.state = ReceiverState.WAITING
        self.ctx = FSMContext()
    
    def reset(self):
        """Reset FSM to initial state."""
        self.state = ReceiverState.WAITING
        self.ctx = FSMContext()
    
    def run(self) -> bool:
        """
        Run the FSM until it reaches DONE state.
        Returns True if processing completed, False if shutting down.
        """
        while self.state != ReceiverState.DONE:
            self._transition()
        return True
    
    def _transition(self):
        """Execute current state and transition to next."""
        match self.state:
            case ReceiverState.WAITING:
                self._state_waiting()
            case ReceiverState.VALIDATING:
                self._state_validating()
            case ReceiverState.RESTART:
                self._state_restart()
            case ReceiverState.PREFILL_CHUNK:
                self._state_prefill_chunk()
            case ReceiverState.OUTPUT:
                self._state_output()
            case ReceiverState.PROCESS_LAYERS:
                self._state_process_layers()
            case ReceiverState.SEND:
                self._state_send()
    
    def _state_waiting(self):
        """Wait for a job from the queue."""
        layer_job = self.receiver._wait_for_job()
        if layer_job is None:
            # Shutting down
            self.state = ReceiverState.DONE
            return
        
        self.ctx.layer_job = layer_job
        self.state = ReceiverState.VALIDATING
    
    def _state_validating(self):
        """Validate pipe availability and gather resources."""
        layer_job = self.ctx.layer_job
        if layer_job is None:
            self.state = ReceiverState.DONE
            return
        
        self.ctx.pipe = self.receiver.job_manager.get_pipe(layer_job.pipe_id)
        
        # Pipe unavailable - drop job
        if self.ctx.pipe is None or not self.ctx.pipe.is_complete():
            self.receiver.router.logger.warning(f"Pipe not found for {layer_job.job_id}")
            self.state = ReceiverState.DONE
            return
        
        # Determine next state based on job flags
        if layer_job.restart:
            self.state = ReceiverState.RESTART
        elif layer_job.done:
            # Ensure we only process the ends of jobs we sent out
            if layer_job.origin_node_id != self.receiver.router.config.node_id:
                self.state = ReceiverState.DONE
                return
            
            # Ensure we have the end model ready
            self.ctx.end_model = self.receiver.get_end_model(self.ctx.pipe.model_id)
            if self.ctx.end_model is None:
                self.state = ReceiverState.DONE
                return
            
            # Job returned from network - check pending job
            self.ctx.pending_job = self.receiver.job_tracker.get_pending_job(layer_job.job_id)
            if self.ctx.pending_job is None:
                self.receiver.router.logger.warning(f"Pending job not found for {layer_job.job_id}")
                self.state = ReceiverState.DONE
                return
            
            job = self.ctx.pending_job.job
            job.data = layer_job.data

            # Prefill chunking: more chunks?
            if job.current_token == 0 and self.ctx.pending_job.chunking.has_more():
                self.state = ReceiverState.PREFILL_CHUNK
            else:
                self.state = ReceiverState.OUTPUT
        else:
            self.state = ReceiverState.PROCESS_LAYERS
    
    def _state_restart(self):
        """Handle a restart request."""
        layer_job = self.ctx.layer_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        
        if layer_job is None or pipe is None or end_model is None:
            self.state = ReceiverState.DONE
            return
        
        pending_job = self.receiver.job_tracker.get_pending_job(layer_job.job_id)
        if pending_job is None:
            self.receiver.router.logger.warning(f"Pending job not found for restart: {layer_job.job_id}")
            self.state = ReceiverState.DONE
            return
        
        self.ctx.pending_job = pending_job
        job = pending_job.job
        job.current_step = ComputeStep.EMBED
        
        # Re-embed and create new layer job
        self.ctx.layer_job = self._embed_and_create_layer_job(end_model, pending_job)
        
        # Check first layer
        model = pipe.get_layer(0, False)
        if model is None:
            self.state = ReceiverState.DONE
            return
        
        if model.virtual:
            self.state = ReceiverState.SEND
        else:
            self.state = ReceiverState.PROCESS_LAYERS

    def _state_prefill_chunk(self):
        """Handle the next prefill chunk."""
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        
        if layer_job is None or pending_job is None or pipe is None or end_model is None:
            self.state = ReceiverState.DONE
            return
        
        job = pending_job.job
        
        # Update job time to prevent stale timeout during prefill
        pending_job.set_last_update()
        
        # Log chunk completion
        chunk_time_ms = (time() - job.chunk_start_time) * 1000
        self.receiver.router.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
        )
        
        pending_job.chunking.advance()
        chunk_start, chunk_end = pending_job.chunking.get_range()
        
        # Log next chunk start
        job.chunk_start_time = time()
        self.receiver.router.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
        )
        
        job.current_step = ComputeStep.EMBED
        self.ctx.layer_job = self._embed_and_create_layer_job(
            end_model, pending_job, chunk_start, chunk_end
        )
        self.ctx.layer_job.done = False

        job.delta = ""
        if not pipe.send_job_update(job):
            self.state = ReceiverState.DONE
            return
        
        model = pipe.get_layer(self.ctx.layer_job.current_layer, False)
        if model is None:
            self.state = ReceiverState.DONE
            return
        
        if model.virtual:
            self.state = ReceiverState.SEND
        else:
            self.state = ReceiverState.PROCESS_LAYERS
    
    def _state_output(self):
        """Handle norm/head computation and prepare next token."""
        layer_job = self.ctx.layer_job
        pending_job = self.ctx.pending_job
        pipe = self.ctx.pipe
        end_model = self.ctx.end_model
        
        if layer_job is None or pending_job is None or pipe is None or end_model is None:
            self.state = ReceiverState.DONE
            return
        
        job = pending_job.job
        
        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            if pending_job.chunking.is_active():
                chunk_time_ms = (time() - job.chunk_start_time) * 1000
                self.receiver.router.logger.info(
                    f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
                    f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
                )
            
            total_prefill_ms = (time() - job.prefill_start_time) * 1000
            tokens_per_sec = (job.prompt_tokens / total_prefill_ms) * 1000 if total_prefill_ms > 0 else 0
            self.receiver.router.logger.info(
                f"[Prefill] job={job.job_id[:8]} finished: "
                f"prompt_tokens={job.prompt_tokens}, "
                f"total_time={total_prefill_ms:.1f}ms, "
                f"throughput={tokens_per_sec:.1f} tok/s"
            )
        
        job.current_step = ComputeStep.NORM
        
        lt = self._create_head_time()
        end_model.compute_norm(job)
        end_model.compute_head(job)
        lt.set_send_time()
        layer_job.times.append(lt)
        
        if self.receiver.print_times:
            layer_job.print_times(self.receiver.router.logger)
        if self.receiver.print_job_data:
            job.print_job(self.receiver.router.logger)
        layer_job.times = []
        
        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            pipe.complete_job(job)
            self.state = ReceiverState.DONE
            return
        
        # More tokens to generate - update and continue
        if not pipe.send_job_update(job):
            self.state = ReceiverState.DONE
            return
        
        # Embed next token (decode phase - no chunking)
        self.ctx.layer_job = self._embed_and_create_layer_job(end_model, pending_job)
        
        model = pipe.get_layer(self.ctx.layer_job.current_layer, False)
        if model is None:
            self.state = ReceiverState.DONE
            return
        
        if model.virtual:
            self.state = ReceiverState.SEND
        else:
            self.state = ReceiverState.PROCESS_LAYERS
    
    def _state_process_layers(self):
        """Process job through local layers."""
        layer_job = self.ctx.layer_job
        pipe = self.ctx.pipe
        
        if layer_job is None or pipe is None:
            self.state = ReceiverState.DONE
            return
        
        pending_job = self.receiver.job_tracker.get_pending_job(layer_job.job_id)
        if pending_job is None:
            pending_job = self.receiver.job_tracker.add_pending_job(layer_job)
        
        self.ctx.pending_job = pending_job
        
        model = pipe.get_layer(layer_job.current_layer, True)
        if model is None:
            self.state = ReceiverState.DONE
            return
        
        model.process_job(layer_job, pending_job.cache)
        
        # Only update pending job time for layer-only nodes (not the origin node)
        if layer_job.origin_node_id != self.receiver.router.config.node_id:
            pending_job.set_last_update()
        
        self.state = ReceiverState.SEND
    
    def _state_send(self):
        """Send job to next destination."""
        layer_job = self.ctx.layer_job
        pipe = self.ctx.pipe
        
        if layer_job is None or pipe is None:
            self.state = ReceiverState.DONE
            return
        
        if layer_job.done:
            pipe.send_job(layer_job, layer_job.origin_node_id)
        else:
            next_model = pipe.get_layer(layer_job.current_layer, False)
            if next_model is not None:
                pipe.send_job(layer_job, next_model.node_id)
        
        self.state = ReceiverState.DONE
    
    def _create_embed_time(self) -> LayerTime:
        return LayerTime(node_id=self.receiver.router.config.node_id, is_embed=True)

    def _create_head_time(self) -> LayerTime:
        """Create a LayerTime for head operations."""
        return LayerTime(node_id=self.receiver.router.config.node_id, is_head=True)

    def _embed_and_create_layer_job(
        self, 
        end_model: EndModel, 
        pending_job: PendingJob,
        chunk_start: int = 0,
        chunk_end: int = -1
    ) -> LayerJob:
        """Compute embedding and create a new LayerJob."""
        lt = self._create_embed_time()
        end_model.compute_embed(pending_job.job, pending_job.cache, chunk_start, chunk_end)
        lt.set_send_time()
        layer_job = pending_job.job.to_layer_job()
        layer_job.times.append(lt)
        return layer_job