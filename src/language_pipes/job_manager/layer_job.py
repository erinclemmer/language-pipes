from distributed_state_network.util.byte_helper import ByteHelper
from language_pipes.job_manager.job_data import JobData

class LayerJob:
    job_id: str
    current_layer: int
    data: JobData

    def __init__(self, job_id: str, current_layer: int, data: JobData):
        self.job_id = job_id
        self.current_layer = current_layer
        self.data = data

    def to_bytes(self):
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_int(self.current_layer)
        bts.write_bytes(self.data.to_bytes())

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)

        job_id = bts.read_string()
        current_layer = bts.read_int()
        job_data = JobData.from_bytes(bts.read_bytes())

        return LayerJob(job_id, current_layer, job_data)