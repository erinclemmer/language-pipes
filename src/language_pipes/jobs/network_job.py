from typing import List, Optional

from language_pipes.util.byte_helper import ByteHelper
from language_pipes.util.enums import ComputeStep
from language_pipes.jobs.job_data import JobData
from language_pipes.jobs.job_time import JobTime

class NetworkJob:
    job_id: str
    pipe_id: str
    origin_node_id: str
    current_layer: int
    compute_step: ComputeStep
    data: Optional[JobData]
    data_hash: bytes
    times: List[JobTime]

    def __init__(
        self, 
        job_id: str, 
        pipe_id: str,
        origin_node_id: str,
        current_layer: int,
        data: Optional[JobData],
        data_hash: bytes,
        compute_step: ComputeStep,
        times: List[JobTime] = []
    ):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.origin_node_id = origin_node_id
        self.current_layer = current_layer
        self.data = data
        self.data_hash = data_hash
        self.compute_step = compute_step
        self.times = times

    def to_bytes(self):
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.origin_node_id)
        bts.write_int(self.current_layer)
        bts.write_int(self.compute_step.value)
        bts.write_bytes(self.data.to_bytes() if self.data is not None else b'')
        bts.write_bytes(self.data_hash)

        bts.write_int(len(self.times))
        for time in self.times:
            bts.write_bytes(time.to_bytes())

        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)

        job_id = bts.read_string()
        pipe_id = bts.read_string()
        origin_node_id = bts.read_string()
        current_layer = bts.read_int()
        step = ComputeStep(bts.read_int())
        job_bytes = bts.read_bytes()
        job_data = JobData.from_bytes(job_bytes) if job_bytes != b'' else None
        data_hash = bts.read_bytes()

        valid = True
        if data_hash != b'':
            valid = JobData.validate_state(job_bytes, data_hash)

        times = []
        for _ in range(0, bts.read_int()):
            times.append(JobTime.from_bytes(bts.read_bytes()))

        return NetworkJob(
            job_id=job_id, 
            pipe_id=pipe_id, 
            origin_node_id=origin_node_id, 
            current_layer=current_layer, 
            data=job_data, 
            data_hash=data_hash,
            compute_step=step,
            times=times
        ), valid
