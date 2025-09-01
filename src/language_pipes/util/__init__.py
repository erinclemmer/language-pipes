import io
import json
import base64
import ctypes
from uuid import UUID
from hashlib import sha256
from threading import Thread

import torch

def uuid_to_bytes(uid: str) -> bytes:
    return UUID(hex=uid).bytes

def bytes_to_uuid(b: bytes) -> str:
    return str(UUID(bytes=b))

def int_to_bytes(i: int) -> bytes:
    return i.to_bytes(4, 'little', signed=False)

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, 'little', signed=False)

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    bts = io.BytesIO()
    torch.save(t, bts)
    return bts.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    if b == b'':
        return None
    return torch.load(io.BytesIO(b))

def get_tensor_byte_string(t: torch.Tensor) -> str:
    bts = tensor_to_bytes(t)
    return base64.b64encode(bts).decode('utf-8')

def get_hash(data: str) -> str:
    hash = sha256(data.encode())
    return hash.hexdigest()

def get_dict_hash(data: dict):
    return get_hash(json.dumps(data))

def size_of_tensor(t: torch.Tensor):
    return t.element_size() * t.nelement()

def tensor_hash(t: torch.Tensor) -> str:
    return get_hash(get_tensor_byte_string(t))

def stop_thread(thread: Thread):
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
            ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')
