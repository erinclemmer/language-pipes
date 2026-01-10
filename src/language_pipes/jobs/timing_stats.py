from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from language_pipes.jobs.network_job import LayerTime


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


@dataclass
class TimingStats:
    network_ms: List[float] = field(default_factory=list)
    network_pairs_ms: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    embed_ms: List[float] = field(default_factory=list)
    head_ms: List[float] = field(default_factory=list)
    layer_ms: List[float] = field(default_factory=list)
    token_ms: List[float] = field(default_factory=list)

    def add_token(self, layer_times: List[LayerTime]) -> None:
        if not layer_times:
            return
        ordered = sorted(layer_times, key=lambda lt: lt.receive_time)
        for entry in ordered:
            duration_ms = (entry.send_time - entry.receive_time) * 1000.0
            if entry.is_embed:
                self.embed_ms.append(duration_ms)
            elif entry.is_head:
                self.head_ms.append(duration_ms)
            else:
                self.layer_ms.append(duration_ms)

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

    def log_summary(self, logger, job_id: str) -> None:
        def log_line(label: str, stats: Optional[dict]) -> None:
            if stats is None:
                logger.info(f"[Timing] {label}: no samples")
                return
            logger.info(
                f"[Timing] {label}: avg={stats['avg_ms']:.2f}ms "
                f"min={stats['min_ms']:.2f}ms max={stats['max_ms']:.2f}ms "
                f"(n={stats['count']})"
            )

        logger.info(f"[Timing] job={job_id[:8]} summary")
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
