from language_pipes.util.byte_helper import ByteHelper

class JobProgress:
    """How far along a job is: the decode token being generated, or how much of
    the prompt has been prefilled.

    Only the origin advances either one - it owns the tokenizer and the head - so
    every other node on the pipe would otherwise report a job parked at token 0
    for its whole lifetime.

    Deliberately a summary rather than a copy of the origin's ChunkState: how
    prefill is split into chunks is an implementation detail that has no business
    on the wire.
    """

    current_token: int
    prompt_tokens: int
    prefilling: bool
    # Prompt tokens already prefilled; only meaningful while prefilling
    prefill_tokens: int

    def __init__(
        self,
        current_token: int,
        prompt_tokens: int,
        prefilling: bool,
        prefill_tokens: int
    ):
        self.current_token = current_token
        self.prompt_tokens = prompt_tokens
        self.prefilling = prefilling
        self.prefill_tokens = prefill_tokens

    def to_bytes(self) -> bytes:
        bts = ByteHelper()
        bts.write_int(self.current_token)
        bts.write_int(self.prompt_tokens)
        bts.write_int(1 if self.prefilling else 0)
        bts.write_int(self.prefill_tokens)
        return bts.get_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> "JobProgress":
        bts = ByteHelper(data)
        return JobProgress(
            current_token=bts.read_int(),
            prompt_tokens=bts.read_int(),
            prefilling=bts.read_int() == 1,
            prefill_tokens=bts.read_int()
        )
