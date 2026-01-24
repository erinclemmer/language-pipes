from io import BytesIO
from typing import Optional
from distributed_state_network.util import bytes_to_int, int_to_bytes, float_to_bytes, bytes_to_float

class ByteHelper:
    def __init__(self, data: Optional[bytes] = None):
        self.bts = BytesIO(data) if data is not None else BytesIO()

    def write_string(self, s: str):
        encoded = s.encode('utf-8')
        self.write_bytes(encoded)

    def write_int(self, i: int):
        self.bts.write(int_to_bytes(i))

    def write_float(self, f: float):
        self.bts.write(float_to_bytes(f))

    def write_bytes(self, b: bytes):
        self.bts.write(int_to_bytes(len(b)))
        self.bts.write(b)

    def read_string(self):
        return self.read_bytes().decode('utf-8')

    def read_int(self):
        return bytes_to_int(self.bts.read(4))

    def read_float(self):
        return bytes_to_float(self.bts.read(8))

    def read_bytes(self):
        return self.bts.read(bytes_to_int(self.bts.read(4)))

    def get_bytes(self):
        return self.bts.getvalue()
