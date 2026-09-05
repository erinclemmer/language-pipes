"""What this node caches for a job, and under which identity.

The split with the rest of the cache code is by question answered:

- `PromptCache` (`jobs/prompt_cache.py`) owns the entries - one store per node.
- `JobCache` (`jobs/job_cache.py`) is one job's bookkeeping - what it adopted,
  where it is due to be snapshotted. It holds no tensors and talks to no store.
- `CachePolicy` here binds the two to a node: given this node's pipe and end
  model, it decides whether a job may use the cache at all, what identity its
  slice is stored under, and it performs the reads and writes.
- `JobProcessor` decides *when* those moments are - it knows that a pass has
  just finished and that the boundary is therefore covered - and calls in.

Keeping the policy off the state machine is what lets the layer-node paths
(`JobTracker.add_job`, `JobReceiver`) ask the same questions later: they have a
pipe and a node id, but no `JobContext` and no FSM.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from language_pipes.jobs.job import Job
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, MIN_CACHE_TOKENS, PromptCache
from language_pipes.modeling.end_model import EndModel
from language_pipes.pipes.pipe import Pipe


@dataclass
class CacheIdentity:
    """What a stored slice is bound to. A lookup matches on all four."""
    model_id: str
    # Every model process whose layers the slice covers; a reload of any of them
    # invalidates it.
    process_ids: List[str]
    start_layer: int
    end_layer: int


class CachePolicy:
    """One per `JobProcessor`, built from the node's pipe and end model.

    Built with `prompt_cache=None` on a node without a cache, in which case
    every method below is a no-op - so a processor made without one behaves as
    it did before caching existed, and its call sites stay unbranched.
    """

    def __init__(
        self,
        node_id: str,
        pipe: Optional[Pipe],
        end_model: Optional[EndModel],
        prompt_cache: Optional[PromptCache] = None
    ):
        self.node_id = node_id
        self.pipe = pipe
        self.end_model = end_model
        self.prompt_cache = prompt_cache
        self.logger = logging.getLogger(__name__)

    # -- identity ---------------------------------------------------------

    def identity(self, job: Job) -> Optional[CacheIdentity]:
        """What this node's slice of the prefix is, for binding an entry.

        Phase 1 only reuses pipes that live entirely on the origin, so the slice
        is the whole stack and the process ids are the end model's plus every
        local segment's - a reload of any of them has to invalidate the entry.
        `job` is unused while that is true; it is the layer node that will need
        it, to name the segment it hosts.
        """
        end_model = self.end_model
        pipe = self.pipe
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

        return CacheIdentity(pipe.model_id, process_ids, 0, num_hidden_layers - 1)

    def usable(self, job: Job) -> bool:
        """Whether this job may read or write the prompt cache at all."""
        cache = self.prompt_cache
        if cache is None or not job.caching.options.enabled or not cache.enabled():
            return False
        if job.prompt_tokens < MIN_CACHE_TOKENS:
            return False
        end_model = self.end_model
        if end_model is None or self.pipe is None:
            return False
        # Phase 1 has no way to ask another node whether it still holds its
        # slice, so a pipe with any remote segment runs uncached.
        return self.pipe.is_local_to(self.node_id, len(end_model.layers))

    # -- read path --------------------------------------------------------

    def plan(self, job: Job):
        """Adopt the longest usable prefix and decide where to snapshot.

        Runs once, right after tokenize - the first moment the prompt is known -
        and before `init_chunking`, which starts the chunks after whatever was
        adopted.
        """
        cache = self.prompt_cache
        if cache is None or not self.usable(job):
            return

        identity = self.identity(job)
        if identity is None:
            return

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
            identity.model_id, identity.process_ids,
            identity.start_layer, identity.end_layer
        )
        if found is not None:
            blocks, entry = found
            job.cache = cache.adopt(entry)
            job.caching.adopt(blocks)

        # Implicit mode writes at the end of the prompt.
        job.caching.plan_prompt_write(job.prompt_tokens)

    # -- write path -------------------------------------------------------

    def tag_write_point(self, job: Job):
        """Mark the chunk about to be embedded if it ends on a write point.

        Only the prompt is ever tagged here, so a job that has started decoding
        clears its tag and keeps it clear.
        """
        if job.current_token != 0:
            job.caching.clear_pending()
            return
        job.caching.tag(job.chunking.get_range()[1])

    def store_tagged_pass(self, job: Job):
        """Snapshot this node's slice at the tagged boundary."""
        cache = self.prompt_cache
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

        identity = self.identity(job)
        if identity is None:
            return
        cache.store(
            write_id, job.cache, job.origin_node_id, identity.model_id,
            identity.process_ids, identity.start_layer, identity.end_layer,
            tokens, job.caching.options.ttl_seconds
        )

    def store_end_of_response(self, job: Job):
        """Write the prompt-plus-answer entry, which is what makes chat hit.

        The next turn's prompt is this prompt, this answer, and a new user
        message, so an entry stored here is a strict prefix of it.
        """
        cache = self.prompt_cache
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

        identity = self.identity(job)
        if identity is None:
            return
        ids = cache.chain(job.caching.scope, job.input_ids)
        cache.store(
            ids[point // BLOCK_SIZE], job.cache, job.origin_node_id,
            identity.model_id, identity.process_ids,
            identity.start_layer, identity.end_layer, point,
            job.caching.options.ttl_seconds
        )

    # -- reporting --------------------------------------------------------

    def log_fields(self, job: Job) -> str:
        """Cache outcome for the per-job completion line."""
        if self.prompt_cache is None:
            return "cache=off cached=0"
        return job.caching.log_fields()
