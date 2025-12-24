from typing import List
from distributed_state_network.util.byte_helper import ByteHelper
from language_pipes.job_manager.job_data import JobData

class LayerTime:
    receive_time: int
    send_time: int
    start_layer: int
    end_layer: int
    node_id: str

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.node_id)
        bts.write_float(self.receive_time)
        bts.write_float(self.send_time)
        bts.write_int(self.start_layer)
        bts.write_int(self.end_layer)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        lt = LayerTime()
        lt.node_id = bts.read_string()
        lt.receive_time = bts.read_float()
        lt.send_time = bts.read_float()
        lt.start_layer = bts.read_int()
        lt.end_layer = bts.read_int()
        return lt

class LayerJob:
    job_id: str
    pipe_id: str
    origin_node_id: str
    current_layer: int
    done: bool
    data: JobData
    times: List[LayerTime]

    def __init__(
        self, 
        job_id: str, 
        pipe_id: str,
        origin_node_id: str,
        current_layer: int,
        data: JobData,
        done: bool,
        times: List[LayerTime] = []
    ):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.origin_node_id = origin_node_id
        self.current_layer = current_layer
        self.data = data
        self.done = done
        self.times = times

    def to_bytes(self):
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.origin_node_id)
        bts.write_int(self.current_layer)
        bts.write_string("true" if self.done else "false")
        bts.write_bytes(self.data.to_bytes())

        bts.write_int(len(self.times))
        for time in self.times:
            bts.write_bytes(time.to_bytes())

        return bts.get_bytes()

    def set_layer(self, state, current_layer: int):
        self.data.state = state
        self.current_layer = current_layer

    def print_times(self, logger):
        logger.info("Times:")
        for i in range(0, len(self.times)):
            current = self.times[i]
            if i != 0:
                last = self.times[i - 1]
                logger.info(f"[Network] {last.node_id} -> {current.node_id} {(current.receive_time - last.send_time) * 1000.0:.2f}ms")
            lps = (current.end_layer - current.start_layer + 1) / (current.send_time - current.receive_time)
            logger.info(f"[Compute] {current.node_id}: layer {current.start_layer} -> {current.end_layer} ({lps:.2f} layers / sec)")

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)

        job_id = bts.read_string()
        pipe_id = bts.read_string()
        origin_node_id = bts.read_string()
        current_layer = bts.read_int()
        done = bts.read_string() == "true"
        job_data = JobData.from_bytes(bts.read_bytes())

        times = []
        l = bts.read_int()
        for i in range(0, l):
            times.append(LayerTime.from_bytes(bts.read_bytes()))

        times.sort(key=lambda x: x.start_layer)

        return LayerJob(job_id, pipe_id, origin_node_id, current_layer, job_data, done, times)