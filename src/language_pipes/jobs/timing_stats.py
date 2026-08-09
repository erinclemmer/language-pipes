from typing import Optional

from language_pipes.jobs.job_time import JobTime
from language_pipes.jobs.completed_pass import CompletedPass

class TimingData:
    job_id: str
    network_ms: list[float]
    network_pairs_ms: dict[tuple[str, str], list[float]]
    embed_ms: list[float]
    head_ms: list[float]
    layer_ms: list[float]
    token_ms: list[float]
    token_counts: list[int]
    all_times: list[list[JobTime]]

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.all_times = []
        self.network_ms = []
        self.network_pairs_ms = { }
        self.embed_ms = []
        self.head_ms = []
        self.layer_ms = []
        self.token_ms = []
        self.token_counts = []

    def add_times(self, new_times: list[JobTime], token_count: int = 1) -> None:
        if len(new_times) == 0:
            return
        self.all_times.append(new_times)
        ordered = sorted(new_times, key=lambda lt: lt.receive_time)
        for entry in ordered:
            duration_ms = (entry.send_time - entry.receive_time) * 1000.0
            if entry.is_embed:
                self.embed_ms.append(duration_ms)
            elif entry.is_head:
                self.head_ms.append(duration_ms)
            else:
                self.layer_ms.append(duration_ms / (entry.end_layer - entry.start_layer))

        for i in range(1, len(ordered)):
            prev = ordered[i - 1]
            current = ordered[i]
            if prev.node_id == current.node_id:
                continue
            latency_ms = (current.receive_time - prev.send_time) * 1000.0
            if latency_ms >= 0:
                self.network_ms.append(latency_ms)
                key = (prev.node_id, current.node_id)
                self.network_pairs_ms.setdefault(key, []).append(latency_ms)

        token_duration_ms = (ordered[-1].send_time - ordered[0].receive_time) * 1000.0
        if token_duration_ms >= 0:
            self.token_ms.append(token_duration_ms)
            self.token_counts.append(token_count)

    def get_avg_embed_time(self) -> float:
        if len(self.embed_ms) == 0:
            return 0.0
        return sum(self.embed_ms) / len(self.embed_ms)

    def get_avg_layer_time(self):
        if len(self.layer_ms) == 0:
            return 0.0
        return sum(self.layer_ms) / len(self.layer_ms)

    def get_avg_head_time(self):
        if len(self.head_ms) == 0:
            return 0.0
        return sum(self.head_ms) / len(self.head_ms)

    def get_avg_total_time(self):
        if len(self.token_ms) == 0:
            return 0.0
        return sum(self.token_ms) / len(self.token_ms)

    def get_tokens_per_second(self) -> float:
        total_ms = sum(self.token_ms)
        if total_ms <= 0:
            return 0.0
        return sum(self.token_counts) / (total_ms / 1000.0)

class TimingStats:
    output_times: TimingData
    prefill_times: TimingData

    current_times: list[JobTime]

    # Most recently closed pass, either finalized here or handed to us by the
    # origin. Carried onto the next network job so it reaches the whole pipe.
    completed_pass: Optional[CompletedPass]
    # Index of that pass; passes at or below it have already been recorded.
    pass_index: int

    def __init__(self, job_id: str):
        self.output_times = TimingData(job_id)
        self.prefill_times = TimingData(job_id)
        self.current_times = []
        self.completed_pass = None
        self.pass_index = -1

    def add_timing(self, time: JobTime) -> None:
        self.current_times.append(time)

    def add_embed_time(self, node_id: str) -> None:
        self.add_timing(JobTime(node_id=node_id, is_embed=True))

    def add_layer_time(self, node_id: str, start_layer: int, end_layer: int) -> None:
        self.add_timing(JobTime(node_id=node_id, start_layer=start_layer, end_layer=end_layer))

    def add_head_time(self, node_id: str) -> None:
        self.add_timing(JobTime(node_id=node_id, is_head=True))
    
    def set_send_time(self) -> None:
        if len(self.current_times) == 0:
            return
        last_time = self.current_times[-1]
        last_time.set_send_time()
        
    def receive_network_job(self, times: list[JobTime], completed: Optional[CompletedPass] = None) -> None:
        self.current_times = times
        self.record_completed_pass(completed)

    def record_completed_pass(self, completed: Optional[CompletedPass]) -> None:
        """Fold a pass finalized by the origin into this node's stats.

        Skips passes we already have: the origin sees the pass it just closed
        come back on the return trip, and a node hosting two layer ranges is
        handed the same pass twice.
        """
        if completed is None or completed.index <= self.pass_index:
            return
        self.pass_index = completed.index
        self.completed_pass = completed
        times = self.prefill_times if completed.is_prefill else self.output_times
        times.add_times(completed.times, completed.token_count)

    def finalize_token(self) -> None:
        self._finalize(self.output_times, 1, is_prefill=False)

    def finalize_prefill_chunk(self, token_count: int) -> None:
        self._finalize(self.prefill_times, token_count, is_prefill=True)

    def _finalize(self, times: TimingData, token_count: int, is_prefill: bool) -> None:
        pass_times = self.current_times
        self.current_times = []
        times.add_times(pass_times, token_count)
        if len(pass_times) == 0:
            return
        self.pass_index += 1
        self.completed_pass = CompletedPass(
            index=self.pass_index,
            token_count=token_count,
            is_prefill=is_prefill,
            times=pass_times
        )
