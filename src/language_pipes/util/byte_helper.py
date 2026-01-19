from io import BytesIO
import struct


def int_to_bytes(i: int) -> bytes:
    return i.to_bytes(4, "little", signed=False)


def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, "little", signed=False)


def float_to_bytes(f: float) -> bytes:
    return struct.pack(">d", f)


def bytes_to_float(b: bytes) -> float:
    return struct.unpack(">d", b)[0]


class ByteHelper:
    def __init__(self, data: bytes | None = None):
        self.bts = BytesIO(data) if data is not None else BytesIO()

    def write_string(self, s: str):
        encoded = s.encode("utf-8")
        self.write_bytes(encoded)

    def write_int(self, i: int):
        self.bts.write(int_to_bytes(i))

    def write_float(self, f: float):
        self.bts.write(float_to_bytes(f))

    def write_bytes(self, b: bytes):
        self.bts.write(int_to_bytes(len(b)))
        self.bts.write(b)

    def read_string(self):
        return self.read_bytes().decode("utf-8")

    def read_int(self):
        return bytes_to_int(self.bts.read(4))

    def read_float(self):
        return bytes_to_float(self.bts.read(8))

    def read_bytes(self):
        length = bytes_to_int(self.bts.read(4))
        return self.bts.read(length)

    def get_bytes(self):
        return self.bts.getvalue()
