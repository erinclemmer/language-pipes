from time import time
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from promise import Promise
from typing import Callable
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_progress import JobProgress
from language_pipes.jobs.network_job import NetworkJob
from language_pipes.jobs.timing_stats import TimingStats

from language_pipes.util.chat import ChatMessage
from language_pipes.util.chunk_state import ChunkState
from language_pipes.util.enums import ComputeStep, JobStatus

# How many times the origin resends the same pass before it gives up. A packet
# that fails its hash is usually transport corruption and comes good on the
# retry; a node whose serialization is deterministically broken must not loop
# until the stale timer.
MAX_PASS_RETRIES = 3

# What identifies one visit of a pass to this node. A node can host two layer
# ranges of the same pipe, so it can see the same pass twice, with a different
# entry point each time.
PassKey = Tuple[ComputeStep, int]


@dataclass
class SavedPass:
    """The output of one pass, kept so the node can resend it on a restart.

    It holds the whole `JobData`, not only the hidden state: Gemma 4 mutates
    `shared_kv_states` as the pass flows and the next node needs the mutated
    copy. The cost is one pass of hidden state per entry point - tens of KB
    while decoding, up to about a MB for a prefill chunk - bounded by
    `max_node_jobs` and released with the job. `get_job_ram` reports KV cache
    only and deliberately does not count it.
    """
    pass_idx: int
    data: Optional[JobData]
    compute_step: ComputeStep
    current_layer: int


class Job:
    # IDs
    job_id: str
    pipe_id: str
    model_id: str
    origin_node_id: str
    
    # Computed
    delta: str
    prompt_tokens: int = 0

    # State Info
    input_ids: List[int]
    compute_step: ComputeStep
    status: JobStatus
    current_token: int = 0
    current_layer: int = 0
    data: Optional[JobData]
    messages: List[ChatMessage]
    result: Optional[str]
    last_update: float
    timing_stats: TimingStats
    stale: bool
    cancel_reason: Optional[str]
    # Origin's progress report, kept only on the nodes that can't derive it
    reported_progress: Optional[JobProgress]

    # Restart bookkeeping. The origin numbers every pass it dispatches from 1;
    # every node keeps the output of the pass it last handled so a restart is a
    # replay rather than a recompute. See documentation/job-processor.md.
    pass_idx: int
    last_pass_idx: int
    pass_key: Optional[PassKey]
    pass_outputs: Dict[PassKey, SavedPass]
    pass_retries: int
    replaying: bool
    # Set when a packet is refused for a reason the job cannot survive; the
    # receiver turns it into a cancel so the caller gets an error, not a timeout.
    receive_error: Optional[str]
    
    # API params
    top_k: int
    top_p: float
    min_p: float
    temperature: float
    presence_penalty: float
    max_completion_tokens: int

    # Classes
    cache: DynamicCache
    chunking: ChunkState

    # Functions
    resolve: Promise | None
    update: Optional[Callable[["Job"], None]]
    complete: Callable[[], None]

    def __init__(
            self,
            origin_node_id: str,
            messages: List[ChatMessage],
            pipe_id: str,
            model_id: str,
            config: PretrainedConfig,
            data: Optional[JobData] = None,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            max_completion_tokens: int = 1000,
            resolve: Optional[Promise] = None,
            update: Optional[Callable[["Job"], None]] = None,
            complete: Optional[Callable[["Job"], None]] = None
        ):
        self.pipe_id = pipe_id
        self.model_id = model_id
        self.job_id = str(uuid4())
        self.origin_node_id = origin_node_id
        
        self.status = JobStatus.IN_PROGRESS
        self.compute_step = ComputeStep.TOKENIZE

        self.delta = ''
        self.data = data
        self.result = None
        self.input_ids = []
        self.timing_stats = TimingStats(self.job_id)
        self.prompt_tokens = 0
        self.current_token = 0
        self.stale = False
        self.cancel_reason = None
        self.reported_progress = None
        self.messages = messages

        self.pass_idx = 0
        self.last_pass_idx = 0
        self.pass_key = None
        self.pass_outputs = { }
        self.pass_retries = 0
        self.replaying = False
        self.receive_error = None

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_completion_tokens = max_completion_tokens
        
        self.current_layer = 0

        self.cache = DynamicCache(config=config)
        self.chunking = ChunkState(self.job_id)
        self.resolve = resolve
        self.update = update

        if complete is not None:
            self.complete = lambda: complete(self)
        else:
            self.complete = self.pass_complete
        
        self.last_update = time()

    def pass_complete(self):
        pass

    def init_chunking(self):
        self.chunking.init(self.prompt_tokens)

    def past_seen_tokens(self) -> int:
        """Tokens already consumed by the cache, tracked here rather than read back
        from `self.cache`.

        Only the layers this node hosts ever land in `self.cache`, and
        `DynamicCache.get_seq_length` reports through the first *attention* layer,
        which for a hybrid linear-attention stack can live on another node and read
        as 0 forever - Qwen3.5 opens with three `linear_attention` layers, so the
        default single local layer never advances it.
        """
        if self.current_token == 0:
            # Still prefilling: chunks that have already finished.
            return self.chunking.get_tokens_processed()

        # Decoding: every token but the one about to be embedded.
        return len(self.input_ids) - 1

    def set_layer(self, state: torch.Tensor, layer: int, num_hidden_layers: int, shared_kv_states: Optional[dict] = None):
        if self.compute_step != ComputeStep.LAYER:
            raise Exception('Invalid step for layer')
        self.current_layer = layer
        if self.data is None:
            return
        self.data.state = state
        # Gemma4 cross-node KV sharing: persist the mutated dict so it serializes onward.
        if shared_kv_states is not None:
            self.data.shared_kv_states = shared_kv_states
        if self.current_layer == num_hidden_layers:
            self.compute_step = ComputeStep.EMBED if self.chunking.has_more() else ComputeStep.HEAD
            self.current_layer = 0

    def set_norm(self, state: torch.Tensor):
        if self.compute_step != ComputeStep.NORM:
            raise Exception('Invalid step for norm')
        if self.data is None:
            return
        self.data.state = state
        self.next_step()

    def set_output(self, token: int, eos_token: int | Iterable[int] | None):
        if self.compute_step != ComputeStep.HEAD:
            raise Exception('Invalid step for head')
        self.input_ids.append(token)
        self.next_step()

        if eos_token is None:
            return

        stop_tokens = {eos_token} if isinstance(eos_token, int) else set(eos_token)

        if token in stop_tokens:
            self.status = JobStatus.COMPLETED

    def input_id_tensor(self):
        if self.input_ids is None:
            return None
        return torch.tensor(self.input_ids)

    def next_step(self):
        if self.compute_step == ComputeStep.TOKENIZE:
            self.compute_step = ComputeStep.EMBED
        elif self.compute_step == ComputeStep.EMBED:
            self.compute_step = ComputeStep.LAYER
        elif self.compute_step == ComputeStep.LAYER:
            self.compute_step = ComputeStep.NORM
        elif self.compute_step == ComputeStep.NORM:
            self.compute_step = ComputeStep.HEAD
        elif self.current_token < self.max_completion_tokens:
            self.current_token += 1
            self.compute_step = ComputeStep.EMBED
            if self.current_token == self.max_completion_tokens:
                self.status = JobStatus.COMPLETED
        else:
            self.status = JobStatus.COMPLETED

    def start_pass(self):
        """Begin a new pass here. Only the origin starts one; it numbers the
        pass and every node downstream carries that number."""
        self.pass_idx += 1
        self.pass_retries = 0
        self.pass_key = (ComputeStep.EMBED, 0)
        self.replaying = False

    def save_pass_output(self):
        """Keep what this node is forwarding, so a restart can resend it.

        A job is strictly sequential - the origin waits for the pass to come
        back through `HEAD` before it dispatches the next one - so one saved
        pass per entry point answers every restart this node can be asked for.
        """
        if self.pass_key is None:
            return
        self.pass_outputs[self.pass_key] = SavedPass(
            pass_idx=self.pass_idx,
            data=self.data,
            compute_step=self.compute_step,
            current_layer=self.current_layer
        )

    def _replay(self, saved: SavedPass):
        """Put the job back in the state it was in when it sent that pass.

        The cache is not touched: this node already holds the keys and values
        for the pass, and computing it again would append them a second time.
        """
        self.replaying = True
        self.data = saved.data
        self.compute_step = saved.compute_step
        self.current_layer = saved.current_layer

    def _receive_restart(self, network_job: NetworkJob) -> bool:
        """Handle a packet a node bounced back because its hash failed.

        `JobReceiver.restart_token` strips the payload, so a data-less `EMBED`
        packet is the one packet that can only be a restart request. The origin
        must not embed again: `_state_embed` advances the prefill chunk, which
        skips it, and re-embedding a decode token puts it into the caches of
        every node before the corruption point a second time. The origin
        resends the saved pass under the same `pass_idx` instead, and the nodes
        that already computed it replay their own saved output.
        """
        saved = self.pass_outputs.get((ComputeStep.EMBED, 0))
        if saved is None or saved.pass_idx != self.pass_idx:
            return False
        # A peer that predates pass numbering drops the field when it bounces the
        # packet. Only one pass is ever in flight, so an unnumbered restart can
        # only be about that pass.
        if network_job.pass_idx != 0 and network_job.pass_idx != self.pass_idx:
            # A second node bounced the same dead pass; the retry is already out.
            return False

        self.pass_retries += 1
        if self.pass_retries > MAX_PASS_RETRIES:
            self.receive_error = f"packet failed validation after {MAX_PASS_RETRIES} retries"
            return False

        # The pass goes out from the head of the pipe again, so it is that entry
        # point the resend belongs to.
        self.pass_key = (ComputeStep.EMBED, 0)
        self._replay(saved)
        return True

    def _accept_pass(self, network_job: NetworkJob) -> Optional[SavedPass]:
        """Decide what to do with an incoming pass.

        Returns the saved pass to replay, or `None` to compute. Sets
        `receive_error` and returns `None` when the pass cannot be honored at
        all - this node's cache holds a different number of positions than the
        packet expects, and computing would produce garbage.
        """
        self.pass_key = (network_job.compute_step, network_job.current_layer)
        if network_job.pass_idx == 0:
            # Peer does not number passes: no sequence to check.
            return None

        saved = self.pass_outputs.get(self.pass_key)
        if saved is not None and saved.pass_idx == network_job.pass_idx:
            return saved

        # `last_pass_idx == 0` is a node that joins the job at this pass, which
        # is how every layer node starts. After that, `==` is a second visit of
        # the same pass, which a node hosting two layer ranges gets, and `+ 1`
        # is the next pass.
        if self.last_pass_idx == 0 or self.last_pass_idx <= network_job.pass_idx <= self.last_pass_idx + 1:
            self.last_pass_idx = network_job.pass_idx
            return None

        self.receive_error = "pass out of sequence"
        return None

    def receive_network_job(self, network_job: NetworkJob, node_id: str) -> bool:
        self.receive_error = None
        if network_job.job_id != self.job_id or network_job.pipe_id != self.pipe_id:
            return False
        if network_job.origin_node_id != self.origin_node_id:
            return False

        if network_job.data is None and network_job.compute_step == ComputeStep.EMBED:
            return self._receive_restart(network_job)

        saved = self._accept_pass(network_job)
        if self.receive_error is not None:
            return False

        # The origin owns the numbering; it never takes one from the wire.
        if node_id != self.origin_node_id:
            self.pass_idx = network_job.pass_idx

        self.timing_stats.receive_network_job(network_job.times, network_job.completed)
        # Origin keeps its own live state; a peer too old to report leaves the
        # last good reading in place
        if node_id != self.origin_node_id and network_job.progress is not None:
            self.reported_progress = network_job.progress

        if saved is not None:
            self._replay(saved)
            return True

        self.replaying = False
        if network_job.compute_step == ComputeStep.HEAD and self.chunking.has_more():
            self.compute_step = ComputeStep.EMBED
            self.current_layer = 0
        else:
            self.compute_step = network_job.compute_step
            self.current_layer = network_job.current_layer

        self.data = network_job.data

        return True

    def get_progress(self) -> JobProgress:
        """This node's own view of how far the job has got - only meaningful on
        the origin, which is the only node that tokenizes, chunks and decodes."""
        return JobProgress(
            current_token=self.current_token,
            prompt_tokens=self.prompt_tokens,
            prefilling=self.chunking.is_active(),
            prefill_tokens=self.chunking.get_tokens_processed()
        )

    def display_progress(self) -> JobProgress:
        """Progress to report in the UI.

        Nodes hosting only layers never advance the token counter or the chunk
        state, so they show what the origin last told them. Their own
        `current_token`/`chunking` are deliberately left alone: `set_layer` reads
        `chunking.has_more()` to decide whether a finished layer pass goes back to
        the origin, and a mirrored chunk state would misroute it.
        """
        if self.reported_progress is not None:
            return self.reported_progress
        return self.get_progress()

    def send_update(self):
        self.last_update_time = time()
        if self.stale:
            return False
        if self.update is not None:
            return self.update(self)
        return True

    def to_network_job(self) -> NetworkJob:
        data_hash = self.data.hash_state() if self.data is not None else b''
        return NetworkJob(
            job_id=self.job_id, 
            pipe_id=self.pipe_id, 
            origin_node_id=self.origin_node_id, 
            current_layer=self.current_layer, 
            data=self.data, 
            data_hash=data_hash, 
            compute_step=self.compute_step,
            times=list(self.timing_stats.current_times),
            completed=self.timing_stats.completed_pass,
            progress=self.get_progress(),
            pass_idx=self.pass_idx
        )

    def set_last_update(self):
        self.last_update = time()

    def get_job_ram(self) -> float:
        """KV cache held for this job, in GB. The saved pass output
        (`pass_outputs`) is one pass of hidden state and is not counted."""
        total_bytes = 0
        tensors = []
        # Newer transformers: cache.layers is a list of layer objects with keys/values
        if hasattr(self.cache, "layers"):
            for layer in self.cache.layers:
                tensors.append(getattr(layer, "keys", None))
                tensors.append(getattr(layer, "values", None))
        # Older transformers: parallel key_cache / value_cache lists of tensors
        else:
            tensors.extend(getattr(self.cache, "key_cache", []))
            tensors.extend(getattr(self.cache, "value_cache", []))

        for tensor in tensors:
            if tensor is not None:
                total_bytes += tensor.numel() * tensor.element_size()

        # Return in GB to match system RAM reporting elsewhere
        return total_bytes / (1024**3)
