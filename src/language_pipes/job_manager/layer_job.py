from distributed_state_network.util.byte_helper import ByteHelper
from language_pipes.job_manager.job_data import JobData

class LayerJob:
    job_id: str
    pipe_id: str
    origin_node_id: str
    current_layer: int
    done: bool
    data: JobData

    def __init__(
        self, 
        job_id: str, 
        pipe_id: str,
        origin_node_id: str,
        current_layer: int,
        data: JobData,
        done: bool
    ):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.origin_node_id = origin_node_id
        self.current_layer = current_layer
        self.data = data
        self.done = done

    def to_bytes(self):
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.origin_node_id)
        bts.write_int(self.current_layer)
        bts.write_string("true" if self.done else "false")
        bts.write_bytes(self.data.to_bytes())

        return bts.get_bytes()

    def set_layer(self, state, current_layer: int):
        self.data.state = state
        self.current_layer = current_layer

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)

        job_id = bts.read_string()
        pipe_id = bts.read_string()
        origin_node_id = bts.read_string()
        current_layer = bts.read_int()
        done = bts.read_string() == "true"
        job_data = JobData.from_bytes(bts.read_bytes())

        return LayerJob(job_id, pipe_id, origin_node_id, current_layer, job_data, done)