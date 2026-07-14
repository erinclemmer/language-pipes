import json
import ctypes
import struct
import threading
from hashlib import sha256

def int_to_bytes(i: int) -> bytes:
    return i.to_bytes(4, 'little', signed=False)

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, 'little', signed=False)

def float_to_bytes(f: float) -> bytes:
    return struct.pack(">d", f)

def bytes_to_float(b: bytes) -> float:
    return struct.unpack(">d", b)[0]

def get_byte_hash(data: bytes) -> bytes:
    return sha256(data).digest()

def get_hash(data: str) -> bytes:
    return get_byte_hash(data.encode('utf-8'))

def get_dict_hash(data: dict) -> bytes:
    return get_hash(json.dumps(data))

def stop_thread(thread: threading.Thread):
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
            ctypes.py_object(SystemExit))
    if res > 1: # pragma: no cover
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')