from language_pipes.util.byte_helper import ByteHelper

class JobCancel:
    """Tells the node that owns a job to stop waiting on it.

    Sent when the pipe a job is running on can no longer carry it - a model was
    unloaded, or the segment for the next layer left the network. Without it the
    origin sits on the job until the stale timeout fires.
    """
    job_id: str
    pipe_id: str
    reason: str

    def __init__(self, job_id: str, pipe_id: str, reason: str):
        self.job_id = job_id
        self.pipe_id = pipe_id
        self.reason = reason

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_string(self.job_id)
        bts.write_string(self.pipe_id)
        bts.write_string(self.reason)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> 'JobCancel':
        bts = ByteHelper(data)
        return JobCancel(
            job_id=bts.read_string(),
            pipe_id=bts.read_string(),
            reason=bts.read_string()
        )
