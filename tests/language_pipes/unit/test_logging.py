import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.util.logging import (
    RingBufferHandler,
    get_ring_buffer,
    setup_logging,
)


def _record(name: str, msg: str) -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, "path", 1, msg, None, None)


class RingBufferHandlerTests(unittest.TestCase):
    def test_keeps_records_in_order(self):
        handler = RingBufferHandler()
        handler.emit(_record("a", "first"))
        handler.emit(_record("a", "second"))

        self.assertEqual([m for _, m in handler.get()], ["first", "second"])

    def test_evicts_oldest_past_capacity(self):
        handler = RingBufferHandler(capacity=2)
        for i in range(4):
            handler.emit(_record("a", str(i)))

        self.assertEqual([m for _, m in handler.get()], ["2", "3"])

    def test_reading_does_not_consume(self):
        """The old per-class lists were drained on read, so two consumers
        starved each other. Any number of readers must see the same records."""
        handler = RingBufferHandler()
        handler.emit(_record("a", "only"))

        self.assertEqual(len(handler.get()), 1)
        self.assertEqual(len(handler.get()), 1)

    def test_filters_by_logger_prefix(self):
        handler = RingBufferHandler()
        handler.emit(_record("language_pipes.oai_server", "http"))
        handler.emit(_record("distributed_state_network.dsnode", "peer"))

        self.assertEqual(
            [m for _, m in handler.get(prefix="distributed_state_network")],
            ["peer"],
        )

    def test_limit_returns_most_recent(self):
        handler = RingBufferHandler()
        for i in range(5):
            handler.emit(_record("a", str(i)))

        self.assertEqual([m for _, m in handler.get(limit=2)], ["3", "4"])

    def test_records_carry_timestamps(self):
        handler = RingBufferHandler()
        record = _record("a", "msg")
        handler.emit(record)

        self.assertEqual(handler.get()[0][0], record.created)


class SetupLoggingTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in self._handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(self._level)

    def test_writes_a_log_file(self):
        """`language-pipes run` previously produced no log file at all, because
        setup lived in the TUI-only main_menu module."""
        with tempfile.TemporaryDirectory() as tmp:
            log_file = setup_logging(Path(tmp), console=False)
            logging.getLogger("language_pipes.test").info("hello from run")

            for handler in logging.getLogger().handlers:
                handler.flush()

            self.assertTrue(log_file.exists())
            self.assertIn("hello from run", log_file.read_text(encoding="utf-8"))

    def test_creates_missing_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "nested" / "logs"
            log_file = setup_logging(log_dir, console=False)

            self.assertTrue(log_dir.is_dir())
            self.assertEqual(log_file.parent, log_dir)

    def test_records_reach_the_ring_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logging(Path(tmp), console=False)
            logging.getLogger("language_pipes.test").info("buffered")

            messages = [m for _, m in get_ring_buffer().get()]
            self.assertIn("buffered", messages)

    def test_console_handler_only_when_requested(self):
        # Count only handlers this test added: the runner may already have a
        # root StreamHandler (pytest installs one).
        def added_stream_handlers():
            return [
                h for h in logging.getLogger().handlers
                if type(h) is logging.StreamHandler and h not in self._handlers
            ]

        with tempfile.TemporaryDirectory() as tmp:
            setup_logging(Path(tmp), console=False)
            # The TUI must not gain a stdout handler; it would corrupt the frame.
            self.assertEqual(added_stream_handlers(), [])

            setup_logging(Path(tmp), console=True)
            self.assertEqual(len(added_stream_handlers()), 1)


if __name__ == "__main__":
    unittest.main()
