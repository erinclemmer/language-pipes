import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from time import time
from typing import Type, TypeVar
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.request_for_model.rfm_packets import RFMPacket, RFMPacketType
from language_pipes.request_for_model import rfm
from language_pipes.request_for_model.rfm import (
    INACTIVITY_TIMEOUT_SECONDS,
    DoneSendingRFMPacket,
    IHaveModelRFMPacket,
    ModelFileData,
    ReadyToReceiveRFMPacket,
    RequestForModelHandler,
    SendingDataRFMPacket,
    WhoHasRFMPacket,
    read_packet,
)

T = TypeVar("T", bound=RFMPacket)


logger = logging.getLogger("lp_test")


class PredeterminedModelFileData(ModelFileData):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.file_hash == other
        return super().__eq__(other)


PREDETERMINED_MANIFEST = {
    "weights.bin": PredeterminedModelFileData(
        "weights.bin",
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        1024 * 1024 * 1024,
    )
}


def patch_fetch_manifest(test_case: unittest.TestCase):
    patcher = patch.object(
        RequestForModelHandler,
        "_fetch_manifest",
        return_value=PREDETERMINED_MANIFEST,
    )
    patcher.start()
    test_case.addCleanup(patcher.stop)

def parse(data: bytes, cls: Type[T]) -> T:
    """Decode wire bytes and assert they dispatch to the expected packet type."""
    pkt = read_packet(data)
    assert isinstance(pkt, cls), f"expected {cls.__name__}, got {type(pkt).__name__}"
    return pkt


def make_handler(test_case: unittest.TestCase, node_id="A", peers=None, installed=None):
    """Build a handler with the four injected callables faked.

    Returns (handler, sent) where ``sent`` is a list of (target_node, data)
    tuples captured from every ``send_packet`` call.
    """
    patch_fetch_manifest(test_case)
    
    sent = []
    handler = RequestForModelHandler(
        node_id=node_id,
        get_peers=lambda: list(peers or []),
        send_packet=lambda node, data: sent.append((node, data)),
        get_installed_models=lambda: list(installed or []),
        logger=logger
    )
    return handler, sent


class PacketSerializationTests(unittest.TestCase):
    """Layer 1: each packet round-trips through create() -> read_packet()."""

    def test_who_has_round_trip(self):
        pkt = parse(WhoHasRFMPacket.create("model-1"), WhoHasRFMPacket)
        self.assertEqual(pkt.req_type, RFMPacketType.WHO_HAS_MODEL)
        self.assertEqual(pkt.model_id, "model-1")

    def test_i_have_round_trip(self):
        pkt = parse(IHaveModelRFMPacket.create("model-1"), IHaveModelRFMPacket)
        self.assertEqual(pkt.req_type, RFMPacketType.I_HAVE_MODEL)
        self.assertEqual(pkt.model_id, "model-1")

    def test_ready_to_receive_round_trip(self):
        pkt = parse(
            ReadyToReceiveRFMPacket.create("model-1"), ReadyToReceiveRFMPacket
        )
        self.assertEqual(pkt.req_type, RFMPacketType.READY_TO_RECEIVE)
        self.assertEqual(pkt.model_id, "model-1")

    def test_sending_data_round_trip(self):
        pkt = parse(
            SendingDataRFMPacket.create("model-1", "weights.bin", 3, False, b"payload"),
            SendingDataRFMPacket,
        )
        self.assertEqual(pkt.req_type, RFMPacketType.SENDING_DATA)
        self.assertEqual(pkt.model_id, "model-1")
        self.assertEqual(pkt.file_name, "weights.bin")
        self.assertEqual(pkt.packet_idx, 3)
        self.assertFalse(pkt.file_done)
        self.assertEqual(pkt.packet_data, b"payload")

    def test_sending_data_file_done_flag(self):
        pkt = parse(
            SendingDataRFMPacket.create("model-1", "weights.bin", 9, True, b""),
            SendingDataRFMPacket,
        )
        self.assertTrue(pkt.file_done)
        self.assertEqual(pkt.packet_data, b"")

    def test_done_sending_round_trip(self):
        pkt = parse(DoneSendingRFMPacket.create("model-1"), DoneSendingRFMPacket)
        self.assertEqual(pkt.req_type, RFMPacketType.DONE_SENDING)
        self.assertEqual(pkt.model_id, "model-1")

    def test_wrong_type_fails_assertion(self):
        # A WHO_HAS payload must not parse as another packet type.
        data = WhoHasRFMPacket.create("model-1")
        with self.assertRaises(AssertionError):
            IHaveModelRFMPacket(data)

    def test_bad_protocol_version_fails(self):
        from language_pipes.util.byte_helper import ByteHelper

        bts = ByteHelper()
        bts.write_int(2)  # wrong protocol
        bts.write_int(RFMPacketType.WHO_HAS_MODEL.value)
        with self.assertRaises(AssertionError):
            read_packet(bts.get_bytes())


class RequestStateTests(unittest.TestCase):
    """Layer 2: requester-side state management and status reporting."""

    def test_request_model_broadcasts_to_all_peers(self):
        handler, sent = make_handler(self, node_id="A", peers=["B", "C"])
        handler.request_model("model-1")

        self.assertEqual(handler.request_state.model_id, "model-1")
        self.assertIsNone(handler.request_state.node_id)
        self.assertIsNone(handler.request_state.status)
        assert handler.request_state.completed_files is not None
        self.assertEqual(len(handler.request_state.completed_files), 0)
        self.assertEqual(handler.request_state.file_data, {})
        self.assertIsNotNone(handler.request_state.expected_manifest)
        self.assertIsNotNone(handler.request_state.last_activity)
        self.assertEqual([node for node, _ in sent], ["B", "C"])
        for _, data in sent:
            pkt = parse(data, WhoHasRFMPacket)
            self.assertEqual(pkt.model_id, "model-1")

    def test_request_model_no_ops_while_already_requesting(self):
        handler, sent = make_handler(self, node_id="A", peers=["B"])
        handler.request_model("model-1")
        sent.clear()

        handler.request_model("model-2")
        self.assertEqual(handler.request_state.model_id, "model-1")  # unchanged
        self.assertEqual(sent, [])

    def test_is_downloading(self):
        handler, _ = make_handler(self, peers=["B"])
        self.assertFalse(handler.request_state.active_download())
        handler.request_model("model-1")
        self.assertTrue(handler.request_state.active_download())

    def test_stop_download_on_reset(self):
        handler, _ = make_handler(self)

        handler.request_model("model-1")
        self.assertTrue(handler.request_state.active_download())
        handler.request_state.reset()
        self.assertFalse(handler.request_state.active_download())

    def test_stop_download_clears_state(self):
        handler, _ = make_handler(self, peers=["B"])
        handler.request_model("model-1")
        handler.request_state.reset()
        self.assertIsNone(handler.request_state.model_id)
        self.assertIsNone(handler.request_state.node_id)
        self.assertIsNone(handler.request_state.status)
        self.assertIsNone(handler.request_state.completed_files)
        self.assertIsNone(handler.request_state.expected_manifest)
        self.assertIsNone(handler.request_state.last_activity)
        self.assertIsNone(handler.request_state.file_data)

    def test_download_status_transitions(self):
        handler, _ = make_handler(self, peers=["B"])
        self.assertIsNone(handler.download_status())

        handler.request_model("model-1")
        self.assertEqual(handler.download_status(), "Downloading: 0 of 1 files completed (0%)")

        # Status reports 32MB per buffered packet, so two packets read as 64MB.
        handler.request_state.file_data = {"weights.bin": {"0": b"x", "1": b"y"}}
        self.assertEqual(handler.download_status(), "Downloading: 0 of 1 files completed (0%)")


class WhoHasTests(unittest.TestCase):
    """Layer 2: provider replies to WHO_HAS only when it owns the model."""

    def test_replies_when_model_installed(self):
        handler, sent = make_handler(self, node_id="B", installed=["model-1"])
        handler.receive_data("A", WhoHasRFMPacket.create("model-1"))

        self.assertEqual(len(sent), 1)
        target, data = sent[0]
        self.assertEqual(target, "A")
        pkt = parse(data, IHaveModelRFMPacket)
        self.assertEqual(pkt.model_id, "model-1")

    def test_silent_when_model_not_installed(self):
        handler, sent = make_handler(self, node_id="B", installed=[])
        handler.receive_data("A", WhoHasRFMPacket.create("model-1"))
        self.assertEqual(sent, [])

    def test_send_to_one_node(self):
        handler, sent = make_handler(self, node_id="A", installed=["model-1"])
        handler.receive_data("B", WhoHasRFMPacket.create("model-1"))
        handler.receive_data("C", WhoHasRFMPacket.create("model-1"))
        self.assertEqual(handler.send_state.node_id, "B")
        self.assertEqual(handler.send_state.model_id, "model-1")
        self.assertEqual(len(sent), 1)

class IHaveTests(unittest.TestCase):
    """Layer 2: requester reacts to I_HAVE by asking the owner to send."""

    def test_records_owner_and_requests_send(self):
        handler, sent = make_handler(self, node_id="A", peers=["B"])
        handler.request_model("model-1")
        sent.clear()

        last_activity = handler.request_state.last_activity

        i_have_pkt = IHaveModelRFMPacket.create("model-1")

        handler.receive_data("B", i_have_pkt)

        assert last_activity is not None
        assert handler.request_state.last_activity is not None

        self.assertTrue(handler.request_state.last_activity > last_activity)

        self.assertEqual(handler.request_state.node_id, "B")
        self.assertEqual(len(sent), 1)
        target, data = sent[0]
        self.assertEqual(target, "B")
        pkt = parse(data, ReadyToReceiveRFMPacket)
        self.assertEqual(pkt.model_id, "model-1")

    def test_ignores_second_owner(self):
        # Once a sender is chosen, a second I_HAVE must not re-trigger a send.
        handler, sent = make_handler(self, node_id="A", peers=["B"])
        handler.request_model("model-1")
        handler._handle_i_have("B", parse(IHaveModelRFMPacket.create("model-1"), IHaveModelRFMPacket))
        sent.clear()

        # receive_data swallows the AssertionError (requesting_node already set).
        handler.receive_data("C", IHaveModelRFMPacket.create("model-1"))
        self.assertEqual(handler.request_state.node_id, "B")
        self.assertEqual(sent, [])

    def test_ignores_other_model(self):
        handler, sent = make_handler(self)
        handler.request_model("model-1")
        handler.receive_data("B", IHaveModelRFMPacket.create("model-2"))
        self.assertEqual(len(sent), 0)
        self.assertIsNone(handler.request_state.node_id)
        self.assertEqual(handler.request_state.model_id, "model-1")

    def test_ignores_with_no_request(self):
        handler, sent = make_handler(self)
        handler.receive_data("B", IHaveModelRFMPacket.create("model-1"))
        self.assertEqual(len(sent), 0)
        self.assertIsNone(handler.request_state.model_id)
        self.assertIsNone(handler.request_state.node_id)   

class _ModelDirTestCase(unittest.TestCase):
    """Base that points get_model_dir() at a throwaway temp directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.model_root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        patcher = patch.object(rfm, "get_model_dir", return_value=self.model_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The transfer path sleeps between packets; keep tests instant.
        sleep_patcher = patch.object(rfm, "sleep", lambda *a, **k: None)
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

        # Completing a download computes model metadata from real weight files;
        # stub it out so the protocol tests don't need a valid model on disk.
        metadata_patcher = patch.object(
            rfm.ModelProvider, "get_model_metadata", return_value=None
        )
        metadata_patcher.start()
        self.addCleanup(metadata_patcher.stop)

    def data_dir(self, model_id):
        return self.model_root / model_id / "data"

    def seed_file(self, model_id, file_name, content):
        d = self.data_dir(model_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / file_name).write_bytes(content)


class SendFileTests(_ModelDirTestCase):
    """Layer 2: provider streams a file then signals DONE_SENDING."""

    def test_streams_file_in_packets_then_done(self):
        self.seed_file("model-1", "weights.bin", b"hello world")
        handler, sent = make_handler(self, node_id="B", installed=["model-1"])
        # WHO_HAS normally records the model the provider is about to send.
        handler.send_state.node_id = "A"
        handler.send_state.model_id = "model-1"

        handler._handle_ready_to_receive(
            "A", parse(ReadyToReceiveRFMPacket.create("model-1"), ReadyToReceiveRFMPacket)
        )

        packets = [read_packet(data) for _, data in sent]
        self.assertTrue(all(node == "A" for node, _ in sent))

        data_packets = [p for p in packets if isinstance(p, SendingDataRFMPacket)]
        done_packets = [p for p in packets if isinstance(p, DoneSendingRFMPacket)]

        # One chunk of content, one zero-length terminator, then DONE_SENDING.
        self.assertEqual(data_packets[0].packet_data, b"hello world")
        self.assertFalse(data_packets[0].file_done)
        self.assertTrue(data_packets[1].file_done)
        self.assertEqual(data_packets[1].packet_data, b"")
        self.assertEqual(len(done_packets), 1)

    def test_multi_chunk_file_uses_increasing_indices(self):
        # Regression: each 32MB chunk must get a distinct, increasing index so the
        # receiver can reassemble it (a single chunk earlier overwrote index 0).
        content = (b"A" * (1024 * 1024 * 32)) + (b"B" * 16)
        self.seed_file("model-1", "big.bin", content)
        handler, sent = make_handler(self, node_id="B", installed=["model-1"])
        handler.send_state.node_id = "A"
        handler.send_state.model_id = "model-1"

        handler._handle_ready_to_receive(
            "A", parse(ReadyToReceiveRFMPacket.create("model-1"), ReadyToReceiveRFMPacket)
        )

        data_packets = [
            p
            for _, data in sent
            for p in [read_packet(data)]
            if isinstance(p, SendingDataRFMPacket)
        ]
        indices = [p.packet_idx for p in data_packets]
        self.assertEqual(indices, list(range(len(data_packets))))
        # One full 32MB chunk, a 16-byte remainder, then a zero-length terminator.
        self.assertEqual(len(data_packets), 3)
        self.assertTrue(data_packets[-1].file_done)

    def test_send_failure_resets_sending_state(self):
        # If a packet send raises mid-transfer (the receiver disconnected), the
        # provider must release sending_data/sending_model so it can serve the
        # next request instead of being stuck on the assert in this handler.
        self.seed_file("model-1", "weights.bin", b"hello world")
        handler, _ = make_handler(self, node_id="B", installed=["model-1"])
        handler.send_state.node_id = "A"
        handler.send_state.model_id = "model-1"

        def boom(node, data):
            raise ConnectionError("receiver disconnected")

        handler._send_packet = boom
        handler._handle_ready_to_receive(
            "A", parse(ReadyToReceiveRFMPacket.create("model-1"), ReadyToReceiveRFMPacket)
        )

        self.assertFalse(handler.send_state.sending)
        self.assertIsNone(handler.send_state.model_id)


class ReceiveFileTests(_ModelDirTestCase):
    """Layer 2: requester accumulates packets and writes the completed file."""

    def _arm_requester(self):
        handler, sent = make_handler(self, node_id="A")
        handler.request_state.model_id = "model-1"
        handler.request_state.node_id = "B"
        handler.request_state.file_data = {}
        handler.request_state.expected_manifest = PREDETERMINED_MANIFEST # pyright: ignore[reportAttributeAccessIssue]
        handler.request_state.completed_files = []
        handler.request_state.mark_activity()
        return handler, sent

    def _feed(self, handler, file_name, chunks):
        """Feed data chunks followed by a zero-length terminator packet."""
        idx = 0
        for chunk in chunks:
            handler._handle_sending_data(
                "B",
                parse(
                    SendingDataRFMPacket.create("model-1", file_name, idx, False, chunk),
                    SendingDataRFMPacket,
                )
            )
            idx += 1
        handler._handle_sending_data(
            "B",
            parse(
                SendingDataRFMPacket.create("model-1", file_name, idx, True, b""),
                SendingDataRFMPacket,
            )
        )

    def test_writes_completed_file_and_clears_buffer(self):
        handler, _ = self._arm_requester()
        self._feed(handler, "weights.bin", [b"hello ", b"world"])

        written = self.data_dir("model-1") / "weights.bin"
        self.assertTrue(written.exists())
        self.assertEqual(written.read_bytes(), b"hello world")
        # Buffer for the finished file is dropped.
        assert handler.request_state.file_data is not None
        self.assertNotIn("weights.bin", handler.request_state.file_data)

    def test_partial_file_is_not_written(self):
        handler, _ = self._arm_requester()
        handler._handle_sending_data(
            "B",
            parse(
                SendingDataRFMPacket.create("model-1", "weights.bin", 0, False, b"part"),
                SendingDataRFMPacket,
            )
        )
        self.assertFalse((self.data_dir("model-1") / "weights.bin").exists())
        assert handler.request_state.file_data is not None
        self.assertIn("weights.bin", handler.request_state.file_data)

    def test_rejects_path_traversal_filename(self):
        handler, _ = self._arm_requester()
        # receive_data swallows the AssertionError; nothing should be buffered.
        handler.receive_data(
            "B", SendingDataRFMPacket.create("model-1", "../escape", 0, False, b"x")
        )
        self.assertEqual(handler.request_state.file_data, {})

    def test_done_sending_fires_callback_and_resets(self):
        handler, _ = self._arm_requester()

        handler._handle_done_sending("B", parse(DoneSendingRFMPacket.create("model-1"), DoneSendingRFMPacket))

        self.assertIsNone(handler.request_state.model_id)
        self.assertIsNone(handler.request_state.node_id)
        self.assertIsNone(handler.request_state.file_data)


class IsFileDoneTests(unittest.TestCase):
    """Layer 2: terminator detection scans contiguous indices for EOF."""

    def _handler(self):
        handler, _ = make_handler(self)
        return handler

    def test_true_when_eof_reached(self):
        handler = self._handler()
        handler.request_state.file_data = {"f": {"0": b"a", "1": b"EOF"}}
        self.assertTrue(handler._is_file_done("f"))

    def test_false_when_gap_before_eof(self):
        handler = self._handler()
        handler.request_state.file_data = {"f": {"0": b"a", "2": b"EOF"}}  # missing idx 1
        self.assertFalse(handler._is_file_done("f"))

    def test_false_for_unknown_file(self):
        handler = self._handler()
        handler.request_state.file_data = {}
        self.assertFalse(handler._is_file_done("missing"))


class InactivityWatchdogTests(unittest.TestCase):
    """Layer 2: the watchdog aborts a download that stalls for too long."""

    def test_check_does_nothing_when_not_downloading(self):
        handler, _ = make_handler(self, peers=["B"])
        handler._check_inactivity()
        self.assertIsNone(handler.request_state.status)

    def test_active_download_is_left_alone(self):
        handler, _ = make_handler(self, peers=["B"])
        handler.request_model("model-1")
        # Activity just happened, so the watchdog must not fire.
        handler._check_inactivity()
        self.assertEqual(handler.request_state.model_id, "model-1")
        self.assertIsNone(handler.request_state.status)

    def test_stalled_download_is_reset_with_error(self):
        handler, _ = make_handler(self, peers=["B"])
        handler.request_model("model-1")
        handler.request_state.node_id = "B"
        # Pretend the last activity was longer ago than the timeout allows.
        handler.request_state.last_activity = time() - (INACTIVITY_TIMEOUT_SECONDS + 1)

        handler._check_inactivity()

        self.assertIsNone(handler.request_state.model_id)
        self.assertIsNone(handler.request_state.node_id)
        self.assertIsNone(handler.request_state.file_data)
        # Status carries "ERROR" so the download page can surface the failure.
        assert handler.request_state.status is not None
        self.assertIn("ERROR", handler.request_state.status)

    def test_receiving_data_resets_the_watchdog(self):
        handler, _ = make_handler(self)
        handler.request_state.model_id = "model-1"
        handler.request_state.node_id = "B"
        handler.request_state.file_data = {}
        handler.request_state.expected_manifest = PREDETERMINED_MANIFEST # pyright: ignore[reportAttributeAccessIssue]
        handler.request_state.last_activity = time() - (INACTIVITY_TIMEOUT_SECONDS + 1)

        handler._handle_sending_data(
            "B",
            parse(
                SendingDataRFMPacket.create("model-1", "weights.bin", 0, False, b"x"),
                SendingDataRFMPacket,
            )
        )
        # Activity refreshed the timestamp, so the watchdog now stays quiet.
        handler._check_inactivity()
        self.assertEqual(handler.request_state.model_id, "model-1")
        self.assertIsNone(handler.request_state.status)

    def test_ready_to_download_again_after_reset(self):
        handler, sent = make_handler(self, node_id="A", peers=["B"])
        handler.request_model("model-1")
        handler.request_state.last_activity = time() - (INACTIVITY_TIMEOUT_SECONDS + 1)
        handler._check_inactivity()
        sent.clear()

        # A fresh request must broadcast again, proving the handler is reusable.
        handler.request_model("model-2")
        self.assertEqual(handler.request_state.model_id, "model-2")
        self.assertEqual([node for node, _ in sent], ["B"])


class HandshakeIntegrationTests(_ModelDirTestCase):
    """Layer 3: two handlers wired together complete the full protocol.

    Each handler's send_packet is routed straight into the other handler's
    receive_data, so a single request_model call drives the whole exchange:
    WHO_HAS -> I_HAVE -> READY_TO_RECEIVE -> SENDING_DATA... -> DONE_SENDING.
    """

    def test_full_transfer(self):
        patch_fetch_manifest(self)
        self.seed_file("model-1", "weights.bin", b"hello world")

        requester = RequestForModelHandler(
            node_id="A",
            get_peers=lambda: ["B"],
            send_packet=lambda node, data: provider.receive_data(node, data),
            get_installed_models=lambda: [],
            logger=logger
        )
        provider = RequestForModelHandler(
            node_id="B",
            get_peers=lambda: ["A"],
            send_packet=lambda node, data: requester.receive_data(node, data),
            get_installed_models=lambda: ["model-1"],
            logger=logger
        )

        requester.request_model("model-1")

        # Callback fired and requester state fully reset.
        self.assertIsNone(requester.request_state.model_id)
        self.assertIsNone(requester.request_state.node_id)
        self.assertIsNone(requester.request_state.file_data)

        # The transferred file landed on disk with the original content.
        written = self.data_dir("model-1") / "weights.bin"
        self.assertEqual(written.read_bytes(), b"hello world")


if __name__ == "__main__":
    unittest.main()
