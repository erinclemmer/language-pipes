import logging
from typing import Callable, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass

from language_pipes.jobs.job import Job
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, MIN_CACHE_TOKENS, PromptCache
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
    # Called when the job cannot go any further (a segment it needs left the
    # network). Cancels the job here and tells the origin node to stop waiting.
    on_fail: Optional[Callable[[Job, str], None]] = None
    # None on a node built without a prompt cache; every cache branch below is
    # skipped when it is, so a processor made without one behaves as it did
    # before caching existed.
    prompt_cache: Optional[PromptCache] = None

def should_prefill_chunk(job: Job) -> bool:
    return job.current_token == 0 and job.chunking.has_more()

def get_next_state(ctx: JobContext) -> JobState:
    # A replay resends the output this node already computed for the pass. It
    # must not run any layer again: the keys and values are in the cache
    # already, and computing would append them a second time.
    if ctx.job.passes.replaying:
        return JobState.SEND

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
    
    Transitions that end because a piece of the pipe is gone (no node hosts a
    layer, or the local end model was unloaded) cancel the job on the way to
    DONE rather than dropping it silently - see _fail.

    VALIDATING -> DONE (missing job, or HEAD step off-origin/without end model,
                        or no node hosts the current layer)
    VALIDATING -> SEND (the job is replaying a pass that failed validation
                        downstream, or the EMBED/TOKENIZE step is off-origin, or
                        the current layer is virtual)
    VALIDATING -> HEAD (HEAD step on origin and prefill finished)
    VALIDATING -> EMBED (EMBED/TOKENIZE step on origin, or more prefill chunks)
    VALIDATING -> PROCESS_LAYERS (current layer is local)

    HEAD -> DONE (missing end model, more prefill chunks, job complete,
                  or failed to send update)
    HEAD -> EMBED (more tokens to generate)
    # HEAD only runs on the origin node, which holds the end model, so it never
    # transitions to SEND or PROCESS_LAYERS.

    EMBED -> DONE (missing end model, failed to send update, or no node hosts
                   the next layer)
    EMBED -> SEND (next layer is virtual/remote)
    EMBED -> PROCESS_LAYERS (next layer is local)

    PROCESS_LAYERS -> DONE (no node hosts the current layer)
    PROCESS_LAYERS -> SEND (next layer set is not local, or all layers done off-origin)
    PROCESS_LAYERS -> PROCESS_LAYERS (next layer set is local)
    PROCESS_LAYERS -> HEAD (all layers done on origin and prefill finished)
    PROCESS_LAYERS -> EMBED (all layers done on origin with more prefill chunks)

    SEND -> DONE (handoff complete, or no node hosts the next layer)
    """
    
    state: JobState
    ctx: JobContext
    
    def __init__(self, ctx: JobContext):
        self.state = JobState.VALIDATING
        self.ctx = ctx
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        while self.state != JobState.DONE:
            # A cancel (model unloaded here or upstream) can land mid-run; stop
            # at the state boundary instead of computing against freed tensors.
            if self.ctx.job.cancel_reason is not None:
                self.state = JobState.DONE
                return
            self.state = self._transition()

    def _fail(self, reason: str) -> JobState:
        """End the job because the pipe can no longer carry it."""
        self.logger.info(f"Job {self.ctx.job.job_id[:4]} stopped: {reason}")
        if self.ctx.on_fail is not None:
            self.ctx.on_fail(self.ctx.job, reason)
        return JobState.DONE

    def _next_state(self) -> JobState:
        """get_next_state, treating a missing layer host as a failure."""
        state = get_next_state(self.ctx)
        if state == JobState.DONE:
            return self._fail(f"no node hosts layer {self.ctx.job.current_layer}")
        return state

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
    
    # -- prompt cache ------------------------------------------------------

    def _cache_identity(self) -> Optional[Tuple[str, List[str], int, int]]:
        """What this node's slice of the prefix is, for binding an entry.

        Phase 1 only reuses pipes that live entirely on the origin, so the slice
        is the whole stack and the process ids are the end model's plus every
        local segment's - a reload of any of them has to invalidate the entry.
        """
        end_model = self.ctx.end_model
        pipe = self.ctx.pipe
        if end_model is None or pipe is None:
            return None

        num_hidden_layers = pipe.num_hidden_layers()
        if num_hidden_layers is None:
            return None

        process_ids = [end_model.process_id]
        layer = len(end_model.layers)
        while layer < num_hidden_layers:
            segment = pipe.get_layer(layer)
            if segment is None or segment.end_layer < layer:
                return None
            process_ids.append(segment.process_id)
            layer = segment.end_layer + 1

        return pipe.model_id, process_ids, 0, num_hidden_layers - 1

    def _cache_usable(self, job: Job) -> bool:
        """Whether this job may read or write the prompt cache at all."""
        cache = self.ctx.prompt_cache
        if cache is None or not job.caching.options.enabled or not cache.enabled():
            return False
        if job.prompt_tokens < MIN_CACHE_TOKENS:
            return False
        end_model = self.ctx.end_model
        if end_model is None or self.ctx.pipe is None:
            return False
        # Phase 1 has no way to ask another node whether it still holds its
        # slice, so a pipe with any remote segment runs uncached.
        return self.ctx.pipe.is_local_to(self.ctx.node_id, len(end_model.layers))

    def _plan_cache(self, job: Job):
        """Adopt the longest usable prefix and decide where to snapshot.

        Runs once, right after tokenize - the first moment the prompt is known -
        and before `init_chunking`, which starts the chunks after whatever was
        adopted.
        """
        cache = self.ctx.prompt_cache
        if cache is None or not self._cache_usable(job):
            return

        identity = self._cache_identity()
        if identity is None:
            return
        model_id, process_ids, start_layer, end_layer = identity

        job.caching.ids = cache.chain(job.caching.scope, job.input_ids)

        # At least one token has to be left to embed, so the longest prefix that
        # could be adopted stops one token short of the prompt.
        max_blocks = (job.prompt_tokens - 1) // BLOCK_SIZE

        # An upper bound on what this job will ask the cache to hold: the entry
        # it may reuse stays resident for the job's life, and its own working
        # cache is what gets stored at the write points.
        estimate = max_blocks * BLOCK_SIZE + job.prompt_tokens + job.max_completion_tokens
        if not cache.reserve(job.job_id, estimate):
            self.logger.info(
                f"Job {job.job_id[:4]} running uncached: "
                f"{estimate} tokens does not fit the cache budget "
                f"(scope {job.caching.scope[:4].hex()})"
            )
            job.caching.forget()
            return
        job.caching.reserved = True

        found = cache.find_longest(
            job.caching.ids, max_blocks, job.origin_node_id,
            model_id, process_ids, start_layer, end_layer
        )
        if found is not None:
            blocks, entry = found
            job.cache = cache.adopt(entry)
            job.caching.adopt(blocks)

        # Implicit mode writes at the end of the prompt.
        job.caching.plan_prompt_write(job.prompt_tokens)

    def _tag_write_point(self, job: Job):
        """Mark the chunk about to be embedded if it ends on a write point.

        Only the prompt is ever tagged here, so a job that has started decoding
        clears its tag and keeps it clear.
        """
        if job.current_token != 0:
            job.caching.clear_pending()
            return
        job.caching.tag(job.chunking.get_range()[1])

    def _store_tagged_pass(self, job: Job):
        """Snapshot this node's slice at the tagged boundary."""
        cache = self.ctx.prompt_cache
        if cache is None:
            return

        pending = job.caching.take_pending()
        if pending is None:
            return
        write_id, tokens = pending

        # A node whose cache has drifted must never write its slice into an
        # entry that later requests will adopt.
        if job.data is None or len(job.data.cache_position) == 0:
            return
        if int(job.data.cache_position[-1]) + 1 != tokens:
            self.logger.debug(
                f"Job {job.job_id[:4]} skipped a cache write at {tokens}: "
                "the pass does not end where the origin said it does"
            )
            return

        identity = self._cache_identity()
        if identity is None:
            return
        model_id, process_ids, start_layer, end_layer = identity
        cache.store(
            write_id, job.cache, job.origin_node_id, model_id,
            process_ids, start_layer, end_layer, tokens,
            job.caching.options.ttl_seconds
        )

    def _store_end_of_response(self, job: Job):
        """Write the prompt-plus-answer entry, which is what makes chat hit.

        The next turn's prompt is this prompt, this answer, and a new user
        message, so an entry stored here is a strict prefix of it.
        """
        cache = self.ctx.prompt_cache
        if cache is None or not job.caching.searched():
            return
        if job.caching.options.mode != "implicit":
            return

        # The last sampled token was never embedded, so the cache covers one
        # position fewer than the sequence.
        covered = len(job.input_ids) - 1
        point = job.caching.response_write_point(covered)
        if point is None:
            return
        if job.data is None or len(job.data.cache_position) == 0:
            return
        if int(job.data.cache_position[-1]) + 1 != covered:
            return

        identity = self._cache_identity()
        if identity is None:
            return
        model_id, process_ids, start_layer, end_layer = identity
        ids = cache.chain(job.caching.scope, job.input_ids)
        cache.store(
            ids[point // BLOCK_SIZE], job.cache, job.origin_node_id, model_id,
            process_ids, start_layer, end_layer, point,
            job.caching.options.ttl_seconds
        )

    def _cache_log_fields(self, job: Job) -> str:
        """Cache outcome for the per-job completion line."""
        if self.ctx.prompt_cache is None:
            return "cache=off cached=0"
        return job.caching.log_fields()

    def _state_validating(self) -> JobState:
        """Validate context for processing"""
        if self.ctx.job is None:
            return JobState.DONE

        # A replaying job only forwards what it already computed, so none of the
        # checks below apply to it.
        if self.ctx.job.passes.replaying:
            return JobState.SEND

        if self.ctx.job.compute_step == ComputeStep.HEAD:
            # Ensure we only process the ends of jobs we sent out
            if self.ctx.job.origin_node_id != self.ctx.node_id:
                return JobState.DONE
            
            # Ensure we have the end model ready
            if self.ctx.end_model is None:
                return self._fail("end model unloaded")

            # Job returned from network - check pending job
            if self.ctx.job is None:
                return JobState.DONE

        return self._next_state()

    def _state_head(self) -> JobState:
        """Handle norm/head computation and prepare to embed the next token."""
        job = self.ctx.job
        end_model = self.ctx.end_model
        if end_model is None:
            return self._fail("end model unloaded")

        # Log prefill completion when transitioning from prefill to decode
        is_prefill = job.current_token == 0
        prefill_chunk_tokens = 0
        if is_prefill:
            if job.chunking.has_more():
                return JobState.DONE

            # Capture the final chunk's length before disable() clears chunk state
            prefill_chunk_tokens = job.chunking.get_chunk_length()
            job.chunking.disable()

        job.compute_step = ComputeStep.NORM
        job.current_layer = 0

        job.timing_stats.add_head_time(self.ctx.node_id)
        end_model.compute_norm(job)
        end_model.compute_head(job)
        job.timing_stats.set_send_time()
        # The pass that produces the first token is still prefill work, so it
        # belongs to the prefill stats rather than the decode averages
        if is_prefill:
            job.timing_stats.finalize_prefill_chunk(prefill_chunk_tokens)
        else:
            job.timing_stats.finalize_token()

        # Job completed
        if job.status == JobStatus.COMPLETED:
            end_model.set_result(job)
            # Before complete(): completing releases the budget reservation this
            # entry is still being charged against.
            self._store_end_of_response(job)
            job.complete()
            self.logger.info(f"Job {job.job_id[:4]} completed {self._cache_log_fields(job)}")
            return JobState.DONE
        
        # More tokens to generate - update and continue
        if not job.send_update():
            job.stale = True
            job.status = JobStatus.COMPLETED
            end_model.set_result(job)
            job.complete()
            self.logger.info(f"Job {job.job_id[:4]} completed {self._cache_log_fields(job)}")
            return JobState.DONE

        return JobState.EMBED

    def _state_embed(self) -> JobState:
        """Embed the next token, handling prefill chunks when needed."""
        job = self.ctx.job
        end_model = self.ctx.end_model

        if end_model is None:
            return self._fail("end model unloaded")

        # Only the origin embeds, and every pass starts here, so this is where
        # the origin gives the pass its number. A restart never reaches this
        # state: it resends the saved pass under the number it already has.
        job.passes.start()

        if job.prompt_tokens == 0:
            end_model.tokenize(job)
            self._plan_cache(job)
            job.init_chunking()
        elif job.chunking.is_active():
            chunk_tokens = job.chunking.get_chunk_length()
            job.chunking.advance()
            job.timing_stats.finalize_prefill_chunk(chunk_tokens)
            job.delta = ""
            if not job.send_update():
                job.stale = True
                job.status = JobStatus.COMPLETED
                end_model.set_result(job)
                job.complete()
                return JobState.DONE

        self._tag_write_point(job)

        job.set_last_update()
        job.timing_stats.add_embed_time(self.ctx.node_id)
        end_model.compute_embed(job)
        job.timing_stats.set_send_time()

        return self._next_state()

    def _state_process_layers(self) -> JobState:
        """Process job through local layers."""
        pipe = self.ctx.pipe
        job = self.ctx.job

        if job.current_layer == 0 and self.ctx.end_model is not None and len(self.ctx.end_model.layers) > 0:
            job.timing_stats.add_layer_time(self.ctx.node_id, 0, len(self.ctx.end_model.layers))
            self.ctx.end_model.compute_layers(job)
            job.timing_stats.set_send_time()

        model = pipe.get_layer(job.current_layer, False)
        if model is None:
            return self._fail(f"no node hosts layer {job.current_layer}")

        if model.virtual:
            return JobState.SEND

        job.timing_stats.add_layer_time(self.ctx.node_id, job.current_layer, model.end_layer)
        model.process_job(job)
        job.timing_stats.set_send_time()
        job.set_last_update()

        # `set_layer` leaves the LAYER step behind once the last layer of the
        # pass is done, which on a Phase 1 pipe is the only moment this node's
        # cache covers exactly the tagged boundary.
        if job.caching.pending_write_id is not None and job.compute_step != ComputeStep.LAYER:
            self._store_tagged_pass(job)

        return self._next_state()

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
                return self._fail(f"no node hosts layer {network_job.current_layer}")
            pipe.send_job(network_job, next_model.node_id)

        # Keep what went out. If a node downstream cannot validate the packet, it
        # bounces back and this is what gets sent again.
        job.passes.save(job.data, job.compute_step, job.current_layer)
        job.passes.sent()

        return JobState.DONE
