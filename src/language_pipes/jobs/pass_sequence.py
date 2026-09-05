from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from language_pipes.jobs.job_data import JobData
from language_pipes.util.enums import ComputeStep

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
    `max_node_jobs` and released with the job. `Job.get_job_ram` reports KV
    cache only and deliberately does not count it.
    """
    pass_idx: int
    data: Optional[JobData]
    compute_step: ComputeStep
    current_layer: int


class PassSequence:
    """Which pass of a job this node is on, and what it sent for the last one.

    A packet that fails its hash is bounced back to the origin, which sends the
    same pass again rather than embedding again. That only works if every node
    can tell a resend from the next pass, so the origin numbers every pass it
    dispatches, from 1, and each node keeps what it forwarded under that number.
    On a resend the node forwards the saved payload untouched: its keys and
    values are in the cache already, and computing again would append them a
    second time.

    A job is strictly sequential - the origin waits for the pass to come back
    through `HEAD` before it dispatches the next - so no node is ever more than
    one pass out of step, and one saved payload per entry point answers every
    resend this node can be asked for.

    `0` means the peer predates pass numbering. Everything then behaves as it
    did before: every packet is computed, and no sequence is checked.

    This object decides; the `Job` that owns it applies the decision to its own
    fields.
    """

    # Which rebuild of the job this is. The origin bumps it when a node reports
    # that it cannot adopt the prefix the job was dispatched with, so every
    # cache on the pipe has to be thrown away and the prefill started again.
    # `(attempt, idx)` totally orders every pass the job has ever sent.
    attempt: int
    # The pass this node is handling. On the origin this is the pass it
    # dispatched; elsewhere it is the number that came in on the wire.
    idx: int
    # The highest pass number seen here, for the sequence check.
    last_idx: int
    # Entry point of the pass being handled, and what was forwarded for each.
    key: Optional[PassKey]
    outputs: Dict[PassKey, SavedPass]
    # Bounces served for the pass in flight. Only the origin counts these.
    retries: int
    # Set while the node is forwarding a saved payload instead of computing.
    replaying: bool
    # Why a packet was refused, when the job cannot survive the refusal. The
    # receiver turns it into a cancel so the caller gets an error, not a timeout.
    error: Optional[str]

    def __init__(self):
        self.attempt = 0
        self.idx = 0
        self.last_idx = 0
        self.key = None
        self.outputs = { }
        self.retries = 0
        self.replaying = False
        self.error = None

    def reset(self, attempt: int):
        """Start the job's numbering over for a new attempt.

        Every saved payload goes with it: they were computed against caches
        that the rebuild is about to discard, so replaying one would forward
        hidden state for a prefix this node no longer holds.
        """
        self.attempt = attempt
        self.idx = 0
        self.last_idx = 0
        self.key = None
        self.outputs = { }
        self.retries = 0
        self.replaying = False
        self.error = None

    def start(self):
        """Number a new pass. Only the origin starts one."""
        self.idx += 1
        self.retries = 0
        self.key = (ComputeStep.EMBED, 0)
        self.replaying = False

    def adopt(self, pass_idx: int):
        """Carry the number the origin gave this pass."""
        self.idx = pass_idx

    def save(self, data: Optional[JobData], compute_step: ComputeStep, current_layer: int):
        """Keep what the node is forwarding, so a restart can resend it."""
        if self.key is None:
            return
        self.outputs[self.key] = SavedPass(
            pass_idx=self.idx,
            data=data,
            compute_step=compute_step,
            current_layer=current_layer
        )

    def sent(self):
        """A pass has gone back out; any replay is over."""
        self.replaying = False

    def restart(self, pass_idx: int) -> Optional[SavedPass]:
        """Answer a bounced packet on the origin.

        Returns the pass to send again, or `None` to drop the packet. The origin
        must not embed again: that advances the prefill chunk, which skips it,
        and puts a decode token into the caches of every node before the point
        of corruption a second time.
        """
        self.error = None
        saved = self.outputs.get((ComputeStep.EMBED, 0))
        if saved is None or saved.pass_idx != self.idx:
            return None
        # A peer that predates pass numbering drops the field when it bounces the
        # packet. Only one pass is ever in flight, so an unnumbered restart can
        # only be about that pass.
        if pass_idx != 0 and pass_idx != self.idx:
            # A second node bounced the same dead pass; the retry is already out.
            return None

        self.retries += 1
        if self.retries > MAX_PASS_RETRIES:
            self.error = f"packet failed validation after {MAX_PASS_RETRIES} retries"
            return None

        # The pass goes out from the head of the pipe again, so it is that entry
        # point the resend belongs to.
        self.key = (ComputeStep.EMBED, 0)
        self.replaying = True
        return saved

    def accept(self, pass_idx: int, compute_step: ComputeStep, current_layer: int) -> Optional[SavedPass]:
        """Decide what to do with an incoming pass.

        Returns the saved pass to forward again, or `None` to compute it. Sets
        `error` when the pass cannot be honored at all - this node's cache holds
        a different number of positions than the packet expects, and computing
        would produce garbage.
        """
        self.error = None
        self.replaying = False
        self.key = (compute_step, current_layer)
        if pass_idx == 0:
            # Peer does not number passes: no sequence to check.
            return None

        saved = self.outputs.get(self.key)
        if saved is not None and saved.pass_idx == pass_idx:
            self.replaying = True
            return saved

        # `last_idx == 0` is a node that joins the job at this pass, which is how
        # every layer node starts. After that, `==` is a second visit of the same
        # pass, which a node hosting two layer ranges gets, and `+ 1` is the next
        # pass.
        if self.last_idx == 0 or self.last_idx <= pass_idx <= self.last_idx + 1:
            self.last_idx = pass_idx
            return None

        self.error = "pass out of sequence"
        return None
