from __future__ import annotations
from logging import Logger
from typing import Dict, Iterable, List, Optional, Tuple

from language_pipes.jobs.job_time import JobTime

def _summary(values: Iterable[float]) -> Optional[dict]:
    values = list(values)
    if not values:
        return None
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }

class TimingData:
    job_id: str
    chunk_size: int
    network_ms: List[float]
    network_pairs_ms: Dict[Tuple[str, str], List[float]]
    embed_ms: List[float]
    head_ms: List[float]
    layer_ms: List[float]
    token_ms: List[float]
    all_times: List[List[JobTime]]

    def __init__(self, job_id: str, chunk_size: int = 1):
        self.job_id = job_id
        self.chunk_size = chunk_size
        self.all_times = []
        self.network_ms = []
        self.network_pairs_ms = { }
        self.embed_ms = []
        self.head_ms = []
        self.layer_ms = []
        self.token_ms = []

    def add_times(self, new_times: List[JobTime]) -> None:
        if len(new_times) == 0:
            return
        self.all_times.append(new_times)
        ordered = sorted(new_times, key=lambda lt: lt.receive_time)
        for entry in ordered:
            duration_ms = ((entry.send_time - entry.receive_time) * 1000.0) / self.chunk_size
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

    def log_summary(self, logger: Logger) -> None:
        def log_line(label: str, stats: Optional[dict]) -> None:
            if stats is None:
                logger.info(f"[Timing] {label}: no samples")
                return
            logger.info(
                f"[Timing] {label}: avg={stats['avg_ms']:.2f}ms "
                f"min={stats['min_ms']:.2f}ms max={stats['max_ms']:.2f}ms "
                f"(n={stats['count']})"
            )

        logger.info(f"[Timing] job={self.job_id[:8]} summary")
        log_line("Network latency", _summary(self.network_ms))
        if self.network_pairs_ms:
            for (source, dest), values in sorted(self.network_pairs_ms.items()):
                stats = _summary(values)
                if stats is None:
                    continue
                logger.info(
                    f"[Timing] Network {source} -> {dest}: avg={stats['avg_ms']:.2f}ms "
                    f"min={stats['min_ms']:.2f}ms max={stats['max_ms']:.2f}ms "
                    f"(n={stats['count']})"
                )
        log_line("Embed", _summary(self.embed_ms))
        log_line("Head", _summary(self.head_ms))
        log_line("Layer", _summary(self.layer_ms))
        log_line("Token", _summary(self.token_ms))

class TimingStats:
    output_times: TimingData
    prefill_times: TimingData
    
    current_times: List[JobTime]

    def __init__(self, job_id: str, prefill_chunk_size: int):
        self.output_times = TimingData(job_id)
        self.prefill_times = TimingData(job_id, prefill_chunk_size)
        self.current_times = []

    def add_timing(self, time: JobTime) -> None:
        self.current_times.append(time)

    def add_embed_time(self, node_id: str) -> None:
        self.add_timing(JobTime(node_id=node_id, is_embed=True))

    def add_layer_time(self, node_id: str, start_layer: int, end_layer: int) -> None:
        self.add_timing(JobTime(node_id=node_id, start_layer=start_layer, end_layer=end_layer))

    def add_head_time(self, node_id: str) -> None:
        self.add_timing(JobTime(node_id=node_id, is_head=True))
    
    def set_send_time(self, logger: Logger) -> None:
        if len(self.current_times) == 0:
            return
        last_time = self.current_times[-1]
        last_time.set_send_time()
        type_str = "LAYER"
        
        if last_time.is_embed:
            type_str = "EMBED"
        
        if last_time.is_head:
            type_str = "HEAD"
        
        elapsed = last_time.send_time - last_time.receive_time
        if type_str == "LAYER":
            elapsed /= last_time.end_layer - last_time.start_layer
        
        logger.info(f"[TIMING] {type_str}: {elapsed*1000.0:.2f}ms")
        
    def receive_network_job(self, times: List[JobTime]) -> None:
        self.current_times = times

    def finalize_token(self, logger: Logger) -> None:
        self.output_times.add_times(self.current_times)
        self.current_times = []
        self.output_times.log_summary(logger)

    def finalize_prefill_chunk(self, logger: Logger) -> None:
        self.prefill_times.add_times(self.current_times)
        self.current_times = []
        self.prefill_times.log_summary(logger)