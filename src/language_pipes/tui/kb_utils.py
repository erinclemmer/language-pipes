import sys
import select
from typing import Optional
from enum import Enum

def key_available():
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(rlist)

class PressedKey(Enum):
    ArrowUp = "ArrowUp"
    ArrowDown = "ArrowDown"
    Enter = "Enter"

def read_key() -> Optional[PressedKey]:
    ch = sys.stdin.read(1)
    if ch == "\n":
        return PressedKey.Enter
    if ch == "\x1b":
        key = sys.stdin.read(2)
        if key == "[A":
            return PressedKey.ArrowUp
        if key == "[B":
            return PressedKey.ArrowDown