import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from language_pipes.util.config import default_log_dir

RING_BUFFER_SIZE = 500

CONSOLE_FORMAT = "%(asctime)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Third-party libraries that attach their own console handler. Name the
# library's root logger: its children own no handlers and propagate up to it.
NOISY_LIBRARIES = ("transformers", )


class QuietLogger(logging.Logger):
    """A logger that refuses handlers and always propagates.

    ``setup_logging`` runs before ``transformers`` is imported -- both entry
    points load it lazily -- and transformers attaches a stderr handler to its
    root logger at import time, with propagation off. Clearing the handlers up
    front would simply be undone, so make the additions no-ops instead: records
    then reach our handlers on the root logger and nothing else.

    ``propagate`` is a data descriptor here, so it wins over the instance
    attribute the library assigns.
    """

    def addHandler(self, hdlr: logging.Handler) -> None:
        pass

    @property
    def propagate(self) -> bool:  # type: ignore[reportIncompatibleVariableOverride]
        return True

    @propagate.setter
    def propagate(self, value: bool) -> None:  # type: ignore[reportIncompatibleVariableOverride]
        pass


def quiet_noisy_libraries():
    """Redirect third-party library output into our handlers.

    Idempotent, and safe to call before the libraries are imported.
    """
    for name in NOISY_LIBRARIES:
        library = logging.getLogger(name)
        for handler in list(library.handlers):
            library.removeHandler(handler)
        library.__class__ = QuietLogger


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the TUI to display.

    Unlike the per-class ``logs`` lists this replaces, reading does not consume:
    any number of consumers can render the same records.
    """

    def __init__(self, capacity: int = RING_BUFFER_SIZE):
        super().__init__()
        self._lock = threading.RLock()
        self.records: Deque[Tuple[float, str, str]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return

        with self._lock:
            self.records.append((record.created, record.name, message))

    def get(self, prefix: Optional[str] = None, limit: Optional[int] = None) -> List[Tuple[float, str]]:
        """Return ``(timestamp, message)`` pairs, oldest first.

        ``prefix`` filters by logger name so a pane can show only its own
        subsystem; ``limit`` keeps only the most recent N.
        """
        with self._lock:
            records = list(self.records)

        if prefix is not None:
            records = [r for r in records if r[1].startswith(prefix)]

        if limit is not None:
            records = records[-limit:]

        return [(created, message) for created, _, message in records]


_ring_buffer: Optional[RingBufferHandler] = None


def get_ring_buffer() -> RingBufferHandler:
    """The in-memory log buffer, created on first use.

    Safe to call before ``setup_logging`` so importers do not depend on
    initialization order.
    """
    global _ring_buffer
    if _ring_buffer is None:
        _ring_buffer = RingBufferHandler()
    return _ring_buffer


def setup_logging(log_dir: Optional[Path] = None, console: bool = False) -> Path:
    """Install the file, ring-buffer, and (optionally) console handlers.

    Called once per process from the CLI, for both the TUI and headless paths.
    ``console`` must stay False under the TUI, where stdout would corrupt the
    ANSI frame.
    """
    if log_dir is None:
        log_dir = default_log_dir()

    log_dir.mkdir(parents=True, exist_ok=True)

    date_suffix = datetime.now().strftime("%d_%m_%Y_%H_%M")
    log_file = log_dir / f"language_pipes_{date_suffix}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_handler)

    root.addHandler(get_ring_buffer())

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        root.addHandler(console_handler)

    quiet_noisy_libraries()

    return log_file
