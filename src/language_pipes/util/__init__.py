import io
import re
import os
import shutil
import struct
from pathlib import Path
import json
import base64
import ctypes
import subprocess
from uuid import UUID
from hashlib import sha256
from threading import Thread
from typing import Optional

import numpy as np
import torch

def uuid_to_bytes(uid: str) -> bytes:
    return UUID(hex=uid).bytes

def bytes_to_uuid(b: bytes) -> str:
    return str(UUID(bytes=b))

def int_to_bytes(i: int) -> bytes:
    return i.to_bytes(4, 'little', signed=False)

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, 'little', signed=False)

# Fast tensor serialization: dtype code (1 byte) + ndim (1 byte) + shape (4 bytes each) + raw data
_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float64: 1, torch.float16: 2,
    torch.int32: 3, torch.int64: 4, torch.int16: 5, torch.int8: 6,
    torch.uint8: 7, torch.bool: 8, torch.bfloat16: 9,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

def tensor_to_bytes(t: torch.Tensor | None) -> bytes:
    if t is None:
        return b''
    t_cpu = t.detach().cpu().contiguous()
    dtype_code = _DTYPE_TO_CODE.get(t_cpu.dtype)
    if dtype_code is None:
        # Fallback for unsupported dtypes
        bts = io.BytesIO()
        torch.save(t_cpu, bts)
        return b'\xff' + bts.getvalue()
    
    shape = t_cpu.shape
    ndim = len(shape)
    header = struct.pack('<BB' + 'I' * ndim, dtype_code, ndim, *shape)
    
    # bfloat16 doesn't have numpy support, so view as int16
    if t_cpu.dtype == torch.bfloat16:
        data = t_cpu.view(torch.int16).numpy().tobytes()
    else:
        data = t_cpu.numpy().tobytes()
    
    return header + data

def bytes_to_tensor(b: bytes) -> torch.Tensor | None:
    if b == b'':
        return None
    
    if b[0] == 0xff:
        # Fallback format
        return torch.load(io.BytesIO(b[1:]), weights_only=True)
    
    dtype_code = b[0]
    ndim = b[1]
    shape = struct.unpack('<' + 'I' * ndim, b[2:2 + 4*ndim])
    dtype = _CODE_TO_DTYPE[dtype_code]
    data = b[2 + 4*ndim:]
    
    if dtype == torch.bfloat16:
        arr = np.frombuffer(data, dtype=np.int16).reshape(shape).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    else:
        np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape).copy()
        return torch.from_numpy(arr)

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

def clone_model(model_id: str, model_dir: str):
    repo_url = f"https://huggingface.co/{model_id}"
    clone_dir = f"{model_dir}/data"

    if not os.path.exists(clone_dir):
        Path(clone_dir).mkdir(parents=True)
    try:
        subprocess.run(["git", "clone", repo_url, clone_dir])
        subprocess.run(["git", "lfs", "install"], cwd=clone_dir, check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=clone_dir, check=True)
    except Exception as e:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        print(e)
        print("Git LFS Error occurred: please ensure that git-lfs is installed")
        exit()


def sanitize_file_name(raw_name: str):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', raw_name).strip().strip('.') + ".toml"

def raise_exception(logger, msg: str):
    logger.exception(msg)
    raise Exception(msg)

def maybeTo(t: Optional[torch.Tensor], device: str) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if str(t.device) == device:
        return t.detach()
    return t.detach().to(device)
