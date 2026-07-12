
from dataclasses import dataclass
import threading
from time import time
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelFileData:
    file_name: str
    file_hash: str
    size: int

class RFMRequestState:
    node_id: Optional[str]
    model_id: Optional[str]
    status: Optional[str]
    completed_files: Optional[List[Tuple[str, int]]]
    lock: threading.RLock
    file_data: Optional[Dict[str, Dict[str, bytes]]]
    expected_manifest: Optional[Dict[str, ModelFileData]]
    last_activity: Optional[float]
    # Files fully received but still being hashed/written to disk on a
    # background thread, and whether DONE_SENDING has arrived. Together these
    # decide when the download can be finalized. Guarded by ``lock``.
    pending_writes: int
    done_received: bool

    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        self.node_id = None
        self.model_id = None
        self.status = None
        self.file_data = None
        self.completed_files = None
        self.expected_manifest = None
        with self.lock:
            self.last_activity = None
            self.pending_writes = 0
            self.done_received = False

    def active_download(self) -> bool:
        return self.model_id is not None and self.last_activity is not None
    
    def inactive_for(self) -> float:
        assert self.last_activity is not None
        return time() - self.last_activity
    
    def mark_activity(self):
        with self.lock:
            self.last_activity = time()

    def start_download(self, model_id: str):
        self.model_id = model_id
        self.file_data = { }
        self.completed_files = []
        self.status = None
        self.mark_activity()

    def total_size(self) -> int:
        assert self.active_download()
        assert self.expected_manifest is not None

        # Bytes, to match downloaded_size(); the two are divided to form the
        # download percentage, so their units must agree.
        total_size = 0
        for key in self.expected_manifest.keys():
            file = self.expected_manifest[key]
            total_size += file.size

        return total_size
    
    def total_file_size(self, file_name: str) -> int:
        assert self.expected_manifest is not None
        assert file_name in self.expected_manifest

        return self.expected_manifest[file_name].size

    def downloaded_file_size(self, file_name: str) -> int:
        assert self.file_data is not None
        if file_name not in self.file_data:
            return 0
        
        total_size = 0
        for pkt_key in self.file_data[file_name].keys():
            total_size += len(self.file_data[file_name][pkt_key])

        return total_size
        
    def downloaded_size(self):
        assert self.active_download()
        assert self.completed_files is not None
        assert self.file_data is not None

        total_size = 0
        for _, size in self.completed_files:
            total_size += size

        for file_key in self.file_data.keys():
            for pkt_key in self.file_data[file_key].keys():
                total_size += len(self.file_data[file_key][pkt_key])

        return total_size
    
    def complete_file(self, file_name: str):
        assert self.active_download()
        assert self.file_data is not None
        assert self.expected_manifest is not None
        assert self.completed_files is not None

        file_size = self.total_file_size(file_name)
        self.completed_files.append((file_name, file_size))
        del self.file_data[file_name]

class RFMSendState:
    node_id: Optional[str]
    model_id: Optional[str]
    sending: bool

    def __init__(self):
        self.reset()

    def reset(self):
        self.sending = False
        self.node_id = None
        self.model_id = None
