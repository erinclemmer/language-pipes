import hashlib
import logging
from logging import Logger
import os
from pathlib import Path
import threading
from time import sleep
from typing import Callable, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from language_pipes.content_provider.model_provider import ModelDownloadProgress, ModelProvider
from language_pipes.request_for_model.rfm_packets import DoneSendingRFMPacket, IHaveModelRFMPacket, ReadyToReceiveRFMPacket, SendingDataRFMPacket, WhoHasRFMPacket
from language_pipes.request_for_model.state import ModelFileData, RFMRequestState, RFMSendState
from language_pipes.request_for_model.util import assert_fn, read_packet
from language_pipes.util.config import get_model_dir

# Skip unauthenticated warning message
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

POLL_INTERVAL_SECONDS = 10
INACTIVITY_TIMEOUT_SECONDS = 30

class RequestForModelHandler:
    request_state: RFMRequestState
    send_state: RFMSendState

    _node_id: str
    _get_peers: Callable[[], List[str]]
    _send_packet: Callable[[str, bytes], None]
    _get_installed_models: Callable[[], List[str]]
    _stop_event: threading.Event
    _monitor_thread: threading.Thread
    _send_thread: Optional[threading.Thread]
    _write_threads: List[threading.Thread]
    _finish_thread: Optional[threading.Thread]

    def __init__(
            self,
            node_id: str,
            get_peers: Callable[[], List[str]],
            send_packet: Callable[[str, bytes], None],
            get_installed_models: Callable[[], List[str]],
            logger: Logger
        ):
        self._node_id = node_id
        self._get_peers = get_peers
        self._send_packet = send_packet
        self._get_installed_models = get_installed_models
        self.request_state = RFMRequestState()
        self.send_state = RFMSendState()
        self.logger = logger
        self._send_thread = None
        self._write_threads = []
        self._finish_thread = None

        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._activity_monitor,
            name=f"rfm-watchdog-{node_id}",
            daemon=True,
        )
        self._monitor_thread.start()

    def _activity_monitor(self):
        while not self._stop_event.wait(POLL_INTERVAL_SECONDS):
            try:
                self._check_inactivity()
            except Exception as e:
                self.logger.error(e)

    def _check_inactivity(self):
        with self.request_state.lock:
            # Don't stop downloading while writing to file
            if self.request_state.pending_writes > 0:
                return
            if not self.request_state.active_download() or self.request_state.inactive_for() < INACTIVITY_TIMEOUT_SECONDS:
                return
            
            self.logger.error(f"Download of {self.request_state.model_id} timed out")
            
            self.request_state.reset()
            self.request_state.status = "ERROR: Download timed out due to inactivity"

    def shutdown(self):
        self._stop_event.set()

    def _download_single_file(self, model_id: str, file_name: str, token: Optional[str]):
        try:
            model_dir = get_model_dir() / model_id / "data"
            hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                token=token,
                local_dir=model_dir,
                tqdm_class=ModelDownloadProgress
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch {file_name} from {model_id}: {e}")
            raise

    def _fetch_manifest(self, model_id: str, token: Optional[str]) -> Optional[Dict[str, ModelFileData]]:
        try:
            info = HfApi().model_info(model_id, files_metadata=True, token=token)
            manifest = { }
            for file in info.siblings or []:
                if file.lfs is None:
                    self._download_single_file(model_id, file.rfilename, token)
                    continue
                manifest[file.rfilename] = ModelFileData(file.rfilename, file.lfs.sha256, file.lfs.size)
            return manifest
        except Exception as e:
            self.logger.error(f"Failed to fetch manifest for {model_id} from HuggingFace: {e}")
            return None

    def request_model(self, model_id: str, token: Optional[str] = None):
        if self.request_state.active_download():
            return

        self.request_state.start_download(model_id)

        manifest = self._fetch_manifest(model_id, token)
        if manifest is None:
            self.request_state.status = "ERROR: Could not fetch model manifest from HuggingFace"
            return
        self.request_state.expected_manifest = manifest
        
        pkt = WhoHasRFMPacket.create(model_id)
        for peer in self._get_peers():
            if peer == self._node_id:
                continue
            self._send_packet(peer, pkt)

    def download_status(self) -> Optional[str]:
        if self.request_state.status is not None:
            return self.request_state.status
        
        if not self.request_state.active_download():
            return None
        
        assert self.request_state.completed_files is not None
        assert self.request_state.file_data is not None
        assert self.request_state.expected_manifest is not None

        downloaded_size = self.request_state.downloaded_size()
        total_size = self.request_state.total_size()
        
        downloaded_files = len(self.request_state.completed_files)
        total_files = len(self.request_state.expected_manifest.keys())

        percent = downloaded_size / total_size * 100 if total_size > 0 else 0
        return f"Downloading: {downloaded_files} of {total_files} files completed ({percent:.0f}%)"

    def receive_data(self, node_id: str, data: bytes):
        try:
            pkt = read_packet(data)
            if isinstance(pkt, WhoHasRFMPacket):
                return self._handle_who_has(node_id, pkt)
            if isinstance(pkt, IHaveModelRFMPacket):
                return self._handle_i_have(node_id, pkt)
            if isinstance(pkt, ReadyToReceiveRFMPacket):
                return self._handle_ready_to_receive(node_id, pkt)
            if isinstance(pkt, SendingDataRFMPacket):
                return self._handle_sending_data(node_id, pkt)
            if isinstance(pkt, DoneSendingRFMPacket):
                return self._handle_done_sending(node_id, pkt)
        except Exception as e:
            self.logger.error(e)
            
    def _handle_who_has(self, node_id: str, pkt: WhoHasRFMPacket):
        assert_fn(self.send_state.model_id is None, "Received WHO_HAS while already sending data")
        assert_fn(not self.send_state.sending, "Cannot handle WHO_HAS while already sending data")
        installed_models = self._get_installed_models()
        if pkt.model_id in installed_models:
            new_pkt = IHaveModelRFMPacket.create(pkt.model_id)
            self.send_state.node_id = node_id
            self.send_state.model_id = pkt.model_id
            self._send_packet(node_id, new_pkt)

    def _handle_i_have(self, node_id: str, pkt: IHaveModelRFMPacket):
        assert_fn(self.request_state.model_id is not None, "Received I_HAVE_MODEL without an active model request")
        assert_fn(self.request_state.node_id is None, "Received I_HAVE_MODEL but already have a responding node")
        assert_fn(pkt.model_id == self.request_state.model_id, "Received I_HAVE packet with wrong model")

        assert self.request_state.model_id is not None
        
        new_pkt = ReadyToReceiveRFMPacket.create(self.request_state.model_id)

        self.request_state.mark_activity()
        self.request_state.node_id = node_id
        self._send_packet(node_id, new_pkt)

    def _handle_ready_to_receive(self, node_id: str, pkt: ReadyToReceiveRFMPacket):
        installed_models = self._get_installed_models()
        assert_fn(node_id == self.send_state.node_id, "Received READY_TO_RECEIVE with node id mismatch")
        assert_fn(pkt.model_id == self.send_state.model_id, "Send state and READY packet model id mismatch")
        assert_fn(pkt.model_id in installed_models, "Requested model is not installed on this node")
        assert_fn(self.send_state.model_id is not None, "Received READY_TO_RECEIVE but no sending model is set")
        assert_fn(not self.send_state.sending, "Received READY_TO_RECEIVE while already sending data")
        
        self.send_state.sending = True

        model_id = pkt.model_id
        def send_worker():
            try:
                self._send_model(node_id)
            except Exception as e:
                self.logger.error(f"Aborting send of {model_id} to {node_id}: {e}")
            finally:
                self.send_state.reset()

        self._send_thread = threading.Thread(
            target=send_worker,
            name=f"rfm-send-{self._node_id}",
            daemon=True,
        )
        self._send_thread.start()
            
    def _send_model(self, node_id: str):
        assert self.send_state.model_id is not None

        model_dir = get_model_dir() / self.send_state.model_id / "data"
        assert_fn(os.path.exists(model_dir), f"Model directory does not exist: {model_dir}")
        for file in list(os.listdir(model_dir)):
            file_path = model_dir / file
            # Skip folders
            if not os.path.isfile(file_path):
                continue
            with open(file_path, 'rb') as f:
                idx = 0
                while True:
                    pkt_data = f.read(1024 * 1024 * 32) # Read 32MB

                    new_pkt = SendingDataRFMPacket.create(
                        self.send_state.model_id,
                        file,
                        idx,
                        len(pkt_data) == 0,
                        pkt_data
                    )

                    self._send_packet(node_id, new_pkt)

                    if len(pkt_data) == 0:
                        break

                    idx += 1

        sleep(1)

        new_pkt = DoneSendingRFMPacket.create(self.send_state.model_id)
        self._send_packet(node_id, new_pkt)

    def _handle_sending_data(self, node_id: str, pkt: SendingDataRFMPacket):
        assert_fn(self.request_state.node_id is not None, "Received SENDING_DATA without an active requesting node")
        assert_fn(node_id == self.request_state.node_id, "SENDING_DATA packet received from unexpected node")
        assert_fn(self.request_state.model_id is not None, "Received SENDING_DATA without an active model request")
        assert_fn(self.request_state.file_data is not None, "Received SENDING_DATA but file_data is not initialized")
        assert_fn(pkt.model_id == self.request_state.model_id, "SENDING_DATA model_id does not match requested model")
        assert self.request_state.model_id is not None
        assert self.request_state.file_data is not None
        assert self.request_state.expected_manifest is not None

        model_dir = get_model_dir() / self.request_state.model_id / "data"
        file_path = model_dir / Path(pkt.file_name)

        # Guard against relative and absolute path injections
        assert_fn(str(model_dir) in str(file_path), "File path escapes the model directory")

        assert_fn(
            pkt.file_name in self.request_state.expected_manifest,
            f"Rejected SENDING_DATA for unexpected file: {pkt.file_name}"
        )

        self.request_state.mark_activity()

        if pkt.file_name not in self.request_state.file_data:
            self.request_state.file_data[pkt.file_name] = { }
        
        assert_fn(str(pkt.packet_idx) not in self.request_state.file_data[pkt.file_name], "Received SENDING_DATA duplicate packet index")

        if not pkt.file_done:
            assert_fn(self.request_state.downloaded_file_size(pkt.file_name) < self.request_state.total_file_size(pkt.file_name), "Received file data overflow from SENDING_DATA packet")
            self.request_state.file_data[pkt.file_name][str(pkt.packet_idx)] = pkt.packet_data
        else:
            self.request_state.file_data[pkt.file_name][str(pkt.packet_idx)] = b'EOF'

        sleep(0.5)

        if self._is_file_done(pkt.file_name):
            with self.request_state.lock:
                self.request_state.pending_writes += 1
            write_thread = threading.Thread(
                target=self._finalize_file,
                args=(pkt.file_name,),
                name=f"rfm-write-{Path(pkt.file_name).name}",
                daemon=True,
            )
            self._write_threads.append(write_thread)
            write_thread.start()

    def _finalize_file(self, file_name: str):
        try:
            self._write_file(file_name)
            self.request_state.complete_file(file_name)
            self.request_state.mark_activity()
        except Exception as e:
            self.logger.error(f"Failed to write {file_name}: {e}")
            self.request_state.reset()
            self.request_state.status = f"ERROR: Failed to write {file_name}"
        finally:
            with self.request_state.lock:
                # reset() zeroes the counter, so clamp instead of going negative.
                self.request_state.pending_writes = max(0, self.request_state.pending_writes - 1)
        self._maybe_finish_download()

    def _is_file_done(self, file_name: str) -> bool:
        if self.request_state.file_data is None or file_name not in self.request_state.file_data:
            return False
        
        file_data = self.request_state.file_data[file_name]
        
        idx = 0
        while True:
            if str(idx) in file_data:
                pkt_data = file_data[str(idx)]
                if pkt_data == b'EOF':
                    return True
            else:
                return False
            idx += 1

    def _write_file(self, file_name: str):
        assert_fn(self.request_state.file_data is not None, "Cannot write file: file_data is None")
        assert_fn(self.request_state.model_id is not None, "Cannot write file: requesting_model is None")
        assert self.request_state.model_id is not None
        assert self.request_state.file_data is not None

        # Reject files not listed in the HuggingFace manifest
        assert_fn(
            self.request_state.expected_manifest is None or file_name in self.request_state.expected_manifest,
            f"File {file_name} is not in the expected model manifest"
        )

        model_dir = get_model_dir() / self.request_state.model_id / "data"
        if not os.path.exists(model_dir):
            model_dir.mkdir(parents=True)

        final_path = model_dir / file_name
        tmp_path = Path(str(final_path) + ".tmp")

        sha256_digest = hashlib.sha256()
        try:
            with open(tmp_path, "wb") as f:
                idx = 0
                file_data = self.request_state.file_data[file_name]
                while True:
                    assert_fn(str(idx) in file_data, f"Missing packet {idx} in file data")
                    file_packet = file_data[str(idx)]
                    if file_packet == b'EOF':
                        break
                    sha256_digest.update(file_packet)
                    f.write(file_packet)
                    idx += 1

            if self.request_state.expected_manifest is not None:
                expected = self.request_state.expected_manifest.get(file_name)
                if expected is not None:
                    actual = sha256_digest.hexdigest()
                    assert_fn(
                        actual == expected.file_hash,
                        f"SHA-256 mismatch for {file_name}: expected {expected.file_hash}, got {actual}"
                    )

            tmp_path.rename(final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _handle_done_sending(self, node_id: str, pkt: DoneSendingRFMPacket):
        assert_fn(self.request_state.node_id is not None, "Received DONE_SENDING without an active requesting node")
        assert_fn(node_id == self.request_state.node_id, "DONE_SENDING packet received from unexpected node")
        assert_fn(self.request_state.model_id is not None, "Received DONE_SENDING without an active model request")
        assert_fn(self.request_state.file_data is not None, "Received DONE_SENDING but file_data is not initialized")
        assert_fn(pkt.model_id == self.request_state.model_id, "Packet model_id and request state model_id mismatch")

        self.request_state.mark_activity()
        with self.request_state.lock:
            self.request_state.done_received = True

        self._finish_thread = threading.Thread(
            target=self._maybe_finish_download,
            name=f"rfm-finish-{self._node_id}",
            daemon=True,
        )
        self._finish_thread.start()

    def _maybe_finish_download(self):
        """Complete the download once DONE_SENDING has arrived and every file
        write has finished; the last of those two events triggers the work."""
        try:
            with self.request_state.lock:
                if not self.request_state.done_received or self.request_state.pending_writes > 0:
                    return
                if not self.request_state.active_download():
                    return

                model_id = self.request_state.model_id
                assert model_id is not None
                assert self.request_state.expected_manifest is not None
                assert self.request_state.completed_files is not None

                received = {name for name, _ in self.request_state.completed_files}
                missing = [f for f in self.request_state.expected_manifest if f not in received]

                # reset() clears done_received, so only one caller gets past here.
                self.request_state.reset()

                if len(missing) > 0:
                    self.logger.error(f"Transfer of {model_id} ended with missing files: {missing}")
                    self.request_state.status = f"ERROR: Transfer ended with {len(missing)} expected file(s) missing"
                    return

                self.request_state.status = "Computing metadata..."

            ModelProvider.get_model_metadata(model_id)
            self.request_state.status = "SUCCESSFULLY Downloaded model"
        except Exception as e:
            self.logger.error(f"Failed to finalize download: {e}")
            self.request_state.reset()
            self.request_state.status = "ERROR: Failed to finalize download"