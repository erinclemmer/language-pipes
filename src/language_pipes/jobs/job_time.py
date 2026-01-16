from time import time
from language_pipes.util.byte_helper import ByteHelper

class JobTime:
    is_embed: bool
    is_head: bool
    receive_time: float
    send_time: float
    start_layer: int
    end_layer: int
    node_id: str

    def __init__(
        self, 
        node_id: str = "",
        is_embed: bool = False, 
        is_head: bool = False,
        start_layer: int = 0,
        end_layer: int = 0
    ):
        self.node_id = node_id
        self.is_embed = is_embed
        self.is_head = is_head
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.receive_time = time()

    def set_send_time(self):
        self.send_time = time()

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.node_id)
        bts.write_int(1 if self.is_embed else 0)
        bts.write_int(1 if self.is_head else 0)
        bts.write_float(self.receive_time)
        bts.write_float(self.send_time)
        bts.write_int(self.start_layer)
        bts.write_int(self.end_layer)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        lt = JobTime()
        lt.node_id = bts.read_string()
        lt.is_embed = bts.read_int() == 1
        lt.is_head = bts.read_int() == 1
        lt.receive_time = bts.read_float()
        lt.send_time = bts.read_float()
        lt.start_layer = bts.read_int()
        lt.end_layer = bts.read_int()
        return lt
