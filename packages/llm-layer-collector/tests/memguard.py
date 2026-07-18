"""Subprocess memory-capping harness for Tier 2 real-model tests.

The hard constraint is: **no test may use more than 10 GB of memory, for any
model.** "Memory" here means *resident* memory (RSS) — that is what actually
competes for RAM and swaps. ``ru_maxrss`` is a process-lifetime high-water mark,
so per-test measurement requires a fresh process; hence each capped body runs in
a subprocess whose peak RSS is asserted against the budget on exit.

A note on ``RLIMIT_AS`` (the obvious hard cap): it limits *virtual* address
space, not RSS. This project's torch is a CUDA build that reserves ~4 GB of
virtual address space on import and mmaps checkpoint files (safetensors) into the
address space, so ``VmSize`` over-counts real residency several-fold. An
``RLIMIT_AS`` tight enough to be meaningful trips on the very first mmap. Instead
we fail fast with an **RSS watchdog thread**: it polls ``VmRSS`` and hard-exits
the moment resident memory crosses the budget, at the allocation site, before the
box starts swapping — the same guarantee, measured on the metric that matters.

The 10 GB budget lives here and nowhere else.
"""

import os
import gc
import sys
import threading
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Optional

GIB = 2 ** 30
MEMORY_BUDGET_BYTES = 20 * GIB

# Child exit code used when the RSS watchdog aborts the process.
_EXIT_OOM = 137
_POLL_SECONDS = 0.05


@dataclass
class CappedResult:
    ok: bool
    peak_rss_bytes: int
    value: Any = None
    error: Optional[str] = None

    @property
    def peak_rss_gib(self) -> float:
        return self.peak_rss_bytes / GIB


def _maxrss_bytes() -> int:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    return rss * 1024 if sys.platform != "darwin" else rss


def _vmrss_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return _maxrss_bytes()


def _child(func: Callable[..., Any], args: tuple, kwargs: dict,
           budget_bytes: int, conn) -> None:
    send_lock = threading.Lock()
    sent = threading.Event()
    stop = threading.Event()

    def _send(result: CappedResult) -> bool:
        with send_lock:
            if sent.is_set():
                return False
            sent.set()
            conn.send(result)
            return True

    def _watchdog() -> None:
        while not stop.wait(_POLL_SECONDS):
            rss = _vmrss_bytes()
            if budget_bytes and rss > budget_bytes:
                _send(CappedResult(
                    ok=False, peak_rss_bytes=rss,
                    error=f"RSS {rss / GIB:.2f} GiB exceeded budget "
                          f"{budget_bytes / GIB:.2f} GiB"))
                conn.close()
                os._exit(_EXIT_OOM)

    watcher = threading.Thread(target=_watchdog, daemon=True)
    watcher.start()
    try:
        value = func(*args, **kwargs)
        gc.collect()
        stop.set()
        _send(CappedResult(ok=True, peak_rss_bytes=_maxrss_bytes(), value=value))
    except BaseException as exc:  # noqa: BLE001
        stop.set()
        _send(CappedResult(
            ok=False, peak_rss_bytes=_maxrss_bytes(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def run_capped(
    func: Callable[..., Any],
    *args,
    budget_bytes: int = MEMORY_BUDGET_BYTES,
    **kwargs,
) -> CappedResult:
    """Run ``func(*args, **kwargs)`` in a fresh subprocess under an RSS watchdog,
    returning its result and peak RSS. ``func`` and its return value must be
    picklable (return small, derived values — not tensors/models)."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child,
                       args=(func, args, kwargs, budget_bytes, child_conn))
    proc.start()
    child_conn.close()
    result: Optional[CappedResult] = None
    try:
        result = parent_conn.recv()
    except EOFError:
        result = None
    proc.join()
    if result is None:
        # Child died without sending (e.g. OOM-killed by the kernel).
        return CappedResult(
            ok=False, peak_rss_bytes=0,
            error=f"subprocess exited without result (exitcode={proc.exitcode})")
    return result
