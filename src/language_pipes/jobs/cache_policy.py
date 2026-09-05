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

Keeping the policy off the state machine is what lets the layer-node paths ask
the same questions: `JobTracker.add_job` adopts a prefix into a job it is still
building, and `JobReceiver` answers the origin - both have a pipe and a node id,
neither has a `JobContext` or an FSM.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from language_pipes.jobs.job import Job
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.prompt_cache import BLOCK_SIZE, MIN_CACHE_TOKENS, PromptCache
from language_pipes.modeling.end_model import EndModel
from language_pipes.pipes.pipe import Pipe
from language_pipes.util.enums import ComputeStep


@dataclass
class CacheIdentity:
    """What a stored slice is bound to. A lookup matches on all four."""
    model_id: str
    # Every model process whose layers the slice covers; a reload of any of them
    # invalidates it.
    process_ids: List[str]
    start_layer: int
    end_layer: int


class CacheOutcome(Enum):
    """What a layer node has to tell the origin after taking on a job.

    Only the receiver can send, so the answer travels back out of `add_job` and
    it does the talking.
    """
    # Adopted what it was told to, and has budget for what it was asked to hold.
    OK = auto()
    # Does not hold the prefix. The job was not added and the pass was not
    # computed; the origin has to rebuild.
    MISS = auto()
    # Adopted and will compute, but has no budget. The origin stops tagging
    # write points so no node stores a prefix this one would be missing.
    NO_STORE = auto()


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

    def local_segments(self) -> List:
        """The physical layer segments of this pipe that run on this node.

        Sorted, so the identity built from them does not depend on the order
        `Pipe.from_meta` happened to assemble them in.
        """
        if self.pipe is None:
            return []
        return sorted(
            [
                s for s in self.pipe.segments
                if s.node_id == self.node_id and not s.virtual and s.loaded
            ],
            key=lambda s: s.start_layer
        )

    def identity(self, job: Job) -> Optional[CacheIdentity]:
        """What this node's slice of the prefix is, for binding an entry.

        The two shapes are different questions, so they are bound differently:

        - On the **origin**, the entry stands for a job running this pipe as it
          is configured now, so every process on the pipe goes into it. A
          segment moving to another node does not corrupt the origin's own
          slice, but it does guarantee the reuse would miss out there, and
          missing locally is cheaper than a rebuild round trip.
        - On a **layer node**, the entry is exactly what its own processes
          computed. It knows nothing about the rest of the pipe and must not
          claim to.

        Either way this is a pure function of the node, the pipe and the job's
        origin, so `add_job` and the write path a pass later agree on it.
        """
        pipe = self.pipe
        if pipe is None:
            return None

        num_hidden_layers = pipe.num_hidden_layers()
        if num_hidden_layers is None:
            return None

        if self.node_id != job.origin_node_id:
            segments = self.local_segments()
            if len(segments) == 0:
                return None
            return CacheIdentity(
                pipe.model_id,
                [s.process_id for s in segments],
                segments[0].start_layer,
                max(s.end_layer for s in segments)
            )

        end_model = self.end_model
        if end_model is None:
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

    def last_local_layer(self, job: Job) -> Optional[int]:
        """The highest layer this node computes for this pipe, or None."""
        ends = [s.end_layer for s in self.local_segments()]
        end_model = self.end_model
        if (
            self.node_id == job.origin_node_id
            and end_model is not None
            and len(end_model.layers) > 0
        ):
            ends.append(len(end_model.layers) - 1)
        return max(ends) if len(ends) > 0 else None

    def pass_complete_here(self, job: Job) -> bool:
        """Whether this node has just finished the last of its layers for the pass.

        That is the one moment its cache covers exactly the tagged boundary and
        no more. A node hosting two ranges of one pipe is visited twice, and a
        snapshot taken after the first visit would be missing this pass for
        every layer of the second.
        """
        if job.compute_step != ComputeStep.LAYER:
            # The pass has left the layer stack altogether.
            return True
        last = self.last_local_layer(job)
        return last is None or job.current_layer > last

    def usable(self, job: Job) -> bool:
        """Whether this job may read or write the prompt cache at all."""
        cache = self.prompt_cache
        if cache is None or not job.caching.options.enabled or not cache.enabled():
            return False
        if job.prompt_tokens < MIN_CACHE_TOKENS:
            return False
        return self.end_model is not None and self.pipe is not None

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
        # Every other node on the pipe has to hold its own slice of whatever
        # this job reuses and stores, so it is told the same estimate.
        job.caching.reserve_tokens = estimate

        found = cache.find_longest(
            job.caching.ids, max_blocks, job.origin_node_id,
            identity.model_id, identity.process_ids,
            identity.start_layer, identity.end_layer
        )
        if found is not None:
            blocks, entry = found
            job.cache = cache.adopt(entry)
            job.caching.adopt(blocks)
            # What the first pass asks the rest of the pipe to adopt. Their
            # slices were stored under the same ID, against their own identity.
            job.caching.use_id = job.caching.ids[blocks]
            job.caching.use_tokens = blocks * BLOCK_SIZE

        # Implicit mode writes at the end of the prompt.
        job.caching.plan_prompt_write(job.prompt_tokens)

    def adopt_for_node(self, job: Job, network_job: NetworkJob) -> CacheOutcome:
        """A layer node's whole read path, run as its `Job` is created.

        It never sees tokens, so it does not search: the origin names one entry
        and this node either holds it or does not. Refusing to compute on a miss
        is what keeps the design honest - a node that quietly prefilled from
        token 0 while the rest of the pipe resumed from 4000 would produce
        wrong output rather than a slow request.
        """
        wants_read = network_job.cache_use_id != b''
        wants_write = network_job.cache_reserve_tokens > 0
        if not wants_read and not wants_write:
            return CacheOutcome.OK

        cache = self.prompt_cache
        identity = None if cache is None else self.identity(job)
        if cache is None or not cache.enabled() or identity is None:
            # Caching is off here, or this node hosts nothing of this pipe.
            return CacheOutcome.MISS if wants_read else CacheOutcome.NO_STORE

        if wants_read:
            entry = cache.lookup(
                network_job.cache_use_id, job.origin_node_id, identity.model_id,
                identity.process_ids, identity.start_layer, identity.end_layer
            )
            # A count that disagrees with the origin's would resume the job at
            # the wrong position, which is worse than not reusing at all.
            hit = entry is not None and entry.token_count == network_job.cache_use_tokens
            cache.count_lookup(hit)
            if not hit:
                return CacheOutcome.MISS
            assert entry is not None
            job.cache = cache.adopt(entry)
            job.caching.adopt(entry.token_count // BLOCK_SIZE)

        if wants_write and not cache.reserve(job.job_id, network_job.cache_reserve_tokens):
            return CacheOutcome.NO_STORE
        job.caching.reserved = wants_write
        return CacheOutcome.OK

    # -- write path -------------------------------------------------------

    def tag_write_point(self, job: Job):
        """Mark the pass about to be embedded if it lands on a write point.

        Every node on the pipe has to snapshot the same boundary, and the only
        moment a node's cache covers exactly that many positions is the end of
        the pass whose last token is the boundary. So the tag is decided here,
        before the embed, and rides the packet to everyone else.

        While decoding, the pass about to run embeds the sequence's last token,
        so afterwards the cache covers `len(input_ids)` positions. A boundary
        landing there is the entry that makes the next turn of a chat hit: the
        next prompt is this prompt, this answer, and one more user message, so
        the entry is a strict prefix of it.
        """
        cache = self.prompt_cache
        if cache is None or not job.caching.searched():
            job.caching.clear_pending()
            return

        if job.current_token == 0:
            job.caching.tag(job.chunking.get_range()[1])
            return

        covered = len(job.input_ids)
        if (
            job.caching.options.mode == "implicit"
            and covered % BLOCK_SIZE == 0
            and job.caching.add_write_point(covered)
        ):
            # The chain has only ever been computed over the prompt. Extending
            # it over the answer names the longer prefix; every ID it shares
            # with the prompt's chain comes out identical, by construction.
            job.caching.ids = cache.chain(job.caching.scope, job.input_ids[:covered])
        job.caching.tag(covered)

    def store_tagged_pass(self, job: Job):
        """Snapshot this node's slice at the tagged boundary."""
        cache = self.prompt_cache
        if cache is None:
            return

        pending = job.caching.pending()
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

    # -- reporting --------------------------------------------------------

    def log_fields(self, job: Job) -> str:
        """Cache outcome for the per-job completion line."""
        if self.prompt_cache is None:
            return "cache=off cached=0"
        return job.caching.log_fields()
