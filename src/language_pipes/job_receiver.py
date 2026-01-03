from time import sleep, time
from threading import Thread
from typing import Callable, Optional, List, Tuple
from distributed_state_network import DSNode

from language_pipes.job_manager.pipe import Pipe
from language_pipes.job_manager.job import Job
from language_pipes.llm_model.end_model import EndModel
from language_pipes.job_manager.enums import ComputeStep, JobStatus
from language_pipes.handlers.job import JobServer
from language_pipes.util import stop_thread
from language_pipes.job_manager.layer_job import LayerJob, LayerTime
from language_pipes.config.processor import ProcessorConfig
from language_pipes.job_manager.pending_job import PendingJob


class JobReceiver:
    port: int
    public_key_file: str
    private_key_file: str
    ecdsa_verification: bool
    print_times: bool
    print_job_data: bool
    prefill_chunk_size: int
    router: DSNode
    job_queue: List[LayerJob]
    get_pipe: Callable[[str], Optional[Pipe]]
    get_end_model: Callable[[str], Optional[EndModel]]
    get_pending_job: Callable[[str], Optional[PendingJob]]
    add_pending_job: Callable[[LayerJob], None]
    update_pending_job: Callable[[str], None]
    restart_job: Callable[[Job], None]

    def __init__(
            self, 
            config: ProcessorConfig,
            router: DSNode,
            get_pipe: Callable[[str], Optional[Pipe]],
            get_end_model: Callable[[str], Optional[EndModel]],
            get_pending_job: Callable[[str], Optional[PendingJob]],
            add_pending_job: Callable[[LayerJob], PendingJob],
            update_pending_job: Callable[[str], None],
            restart_job: Callable[[Job], None]
    ):
        self.router = router
        self.get_pipe = get_pipe
        self.get_end_model = get_end_model
        self.restart_job = restart_job
        self.add_pending_job = add_pending_job
        self.get_pending_job = get_pending_job
        self.update_pending_job = update_pending_job
        self.job_queue = []
        self.ecdsa_verification = config.ecdsa_verification
        self.print_times = config.print_times
        self.print_job_data = config.print_job_data
        self.prefill_chunk_size = config.prefill_chunk_size

        thread, httpd = JobServer.start(config.job_port, self.router, self.receive_data)
        self.thread = thread
        self.httpd = httpd
        Thread(target=self.job_runner, args=()).start()
        router.logger.info(f"Started Job Receiver on port {config.job_port}")

    def _schedule_next(self):
        """Schedule the next job_runner iteration."""
        Thread(target=self.job_runner, args=()).start()

    def _wait_for_job(self) -> Optional[LayerJob]:
        """Wait for a job from the queue. Returns None if shutting down."""
        while True:
            if self.router.shutting_down:
                return None
            if len(self.job_queue) > 0:
                layer_job = self.job_queue.pop()
                return layer_job
            sleep(0.1)

    def _create_embed_time(self) -> LayerTime:
        """Create a LayerTime for embedding operations."""
        return LayerTime(node_id=self.router.config.node_id, is_embed=True)

    def _create_head_time(self) -> LayerTime:
        """Create a LayerTime for head operations."""
        return LayerTime(node_id=self.router.config.node_id, is_head=True)

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
        lt.send_time = time()
        layer_job = pending_job.job.to_layer_job()
        layer_job.times.append(lt)
        return layer_job

    def _handle_restart(
        self, 
        layer_job: LayerJob, 
        pipe: Pipe, 
        end_model: EndModel
    ) -> Tuple[LayerJob, bool, bool]:
        """
        Handle a restart job. Returns (layer_job, should_return, success).
        If should_return is True, caller should return immediately.
        If success is False, the pending job was not found.
        """
        pending_job = self.get_pending_job(layer_job.job_id)
        if pending_job is None:
            return layer_job, True, False
        
        job = pending_job.job
        job.current_step = ComputeStep.EMBED
        
        layer_job = self._embed_and_create_layer_job(end_model, pending_job)
        
        model = pipe.model_for_job(layer_job)
        if model.virtual:
            pipe.send_job(layer_job, model.node_id)
            return layer_job, True, True
        
        return layer_job, False, True

    def _handle_next_prefill_chunk(
        self, 
        layer_job: LayerJob, 
        pending_job: PendingJob,
        pipe: Pipe, 
        end_model: EndModel
    ) -> Tuple[LayerJob, bool]:
        """
        Handle the next prefill chunk. Returns (layer_job, should_return).
        If should_return is True, caller should return immediately.
        """
        job = pending_job.job
        
        # Update job time to prevent stale timeout during prefill
        pending_job.last_update = time()
        
        # Log chunk completion (for the chunk that just returned)
        chunk_time_ms = (time() - job.chunk_start_time) * 1000
        self.router.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
        )
        
        pending_job.chunking.advance()
        chunk_start, chunk_end = pending_job.chunking.get_range()
        
        # Log next chunk start
        job.chunk_start_time = time()
        self.router.logger.info(
            f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
            f"{pending_job.chunking.total_chunks} starting: tokens {chunk_start}-{chunk_end}"
        )
        
        job.current_step = ComputeStep.EMBED
        layer_job = self._embed_and_create_layer_job(
            end_model, pending_job, chunk_start, chunk_end
        )
        layer_job.done = False
        
        model = pipe.model_for_job(layer_job)
        if model.virtual:
            pipe.send_job(layer_job, model.node_id)
            return layer_job, True
        
        return layer_job, False

    def _handle_output_and_next_token(
        self, 
        layer_job: LayerJob, 
        pending_job: PendingJob,
        pipe: Pipe, 
        end_model: EndModel
    ) -> Tuple[LayerJob, bool]:
        """
        Handle norm/head computation and prepare next token if needed.
        Returns (layer_job, should_return). If should_return is True, caller should return.
        """
        job = pending_job.job
        
        # Log prefill completion when transitioning from prefill to decode
        if job.current_token == 0:
            # Log final chunk completion if chunking was active
            if pending_job.chunking.is_active():
                chunk_time_ms = (time() - job.chunk_start_time) * 1000
                self.router.logger.info(
                    f"[Prefill] job={job.job_id[:8]} chunk {pending_job.chunking.current_chunk + 1}/"
                    f"{pending_job.chunking.total_chunks} completed in {chunk_time_ms:.1f}ms"
                )
            
            # Log prefill finished with total time and throughput
            total_prefill_ms = (time() - job.prefill_start_time) * 1000
            tokens_per_sec = (job.prompt_tokens / total_prefill_ms) * 1000 if total_prefill_ms > 0 else 0
            self.router.logger.info(
                f"[Prefill] job={job.job_id[:8]} finished: "
                f"prompt_tokens={job.prompt_tokens}, "
                f"total_time={total_prefill_ms:.1f}ms, "
                f"throughput={tokens_per_sec:.1f} tok/s"
            )
        
        job.current_step = ComputeStep.NORM
        
        lt = self._create_head_time()
        end_model.compute_norm(job)
        end_model.compute_head(job)
        lt.send_time = time()
        layer_job.times.append(lt)
        
        if self.print_times:
            layer_job.print_times(self.router.logger)
        if self.print_job_data:
            job.print_job(self.router.logger)
        layer_job.times = []
        
        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            pipe.complete_job(job)
            return layer_job, True
        
        # More tokens to generate - update and continue
        if not pipe.update_job(job):
            return layer_job, True
        
        # Embed next token (decode phase - no chunking)
        layer_job = self._embed_and_create_layer_job(end_model, pending_job)
        
        # Check if first model is virtual (remote) - if so, send directly
        model = pipe.model_for_job(layer_job)
        if model.virtual:
            pipe.send_job(layer_job, model.node_id)
            return layer_job, True
        
        return layer_job, False

    def _process_and_send(self, layer_job: LayerJob, pipe: Pipe):
        """Process job through local layers and send to next destination."""
        pending_job = self.get_pending_job(layer_job.job_id)
        if pending_job is None:
            pending_job = self.add_pending_job(layer_job)
        
        model = pipe.model_for_job(layer_job)
        model.process_job(layer_job, pending_job.cache)
        
        # Only update pending job time for layer-only nodes (not the origin node)
        # Origin node manages its own job state including current_token
        if layer_job.origin_node_id != self.router.config.node_id:
            self.update_pending_job(layer_job.job_id)

        if layer_job.done:
            pipe.send_job(layer_job, layer_job.origin_node_id)
        else:
            next_model = pipe.model_for_job(layer_job)
            pipe.send_job(layer_job, next_model.node_id)

    def job_runner(self):
        """Main job processing loop."""
        try:
            layer_job = self._wait_for_job()
            if layer_job is None:
                return
            
            pipe = self.get_pipe(layer_job.pipe_id)
            
            # Pipe unavailable - restart job
            if pipe is None or not pipe.is_complete():
                pending_job = self.get_pending_job(layer_job.job_id)
                if pending_job is not None:
                    self.restart_job(pending_job.job)
                self._schedule_next()
                return
            
            end_model = self.get_end_model(pipe.model_id)
            
            # Handle restart request
            if layer_job.restart:
                layer_job, should_return, success = self._handle_restart(layer_job, pipe, end_model)
                if not success:
                    self.router.logger.warning(f"Pending job not found for restart: {layer_job.job_id}")
                if should_return:
                    self._schedule_next()
                    return

            # Handle completed layer processing (job returned from network)
            if layer_job.done:
                pending_job = self.get_pending_job(layer_job.job_id)
                if pending_job is None:
                    # Job not found - may have timed out or been processed elsewhere
                    self.router.logger.warning(f"Pending job not found for {layer_job.job_id}")
                    self._schedule_next()
                    return
                
                job = pending_job.job
                job.data = layer_job.data

                # Prefill chunking: more chunks to process?
                if job.current_token == 0 and pending_job.chunking.has_more():
                    layer_job, should_return = self._handle_next_prefill_chunk(
                        layer_job, pending_job, pipe, end_model
                    )
                    if should_return:
                        self._schedule_next()
                        return
                else:
                    # Final chunk or decode phase - do norm/head
                    layer_job, should_return = self._handle_output_and_next_token(
                        layer_job, pending_job, pipe, end_model
                    )
                    if should_return:
                        self._schedule_next()
                        return

            # Process through local layers and send onward
            self._process_and_send(layer_job, pipe)

        except Exception as e:
            print(e)
        
        self._schedule_next()

    def restart_token(self, job: LayerJob):
        """Mark job for restart and send back to origin."""
        job.restart = True
        pipe = self.get_pipe(job.pipe_id)
        pipe.send_job(job, job.origin_node_id)

    def receive_data(self, data: bytes):
        """Receive and validate incoming job data."""
        job = LayerJob.from_bytes(data)
        
        # Ignore duplicate jobs
        for j in self.job_queue:
            if j.job_id == job.job_id:
                return

        # Validate state integrity
        valid = job.data.validate_state(job.data_hash)
        if not job.restart and (valid is False or valid is None):
            self.restart_token(job)
            return

        self.job_queue.insert(0, job)

    def stop(self):
        """Stop the job receiver."""
        self.httpd.stop()
        stop_thread(self.thread)
