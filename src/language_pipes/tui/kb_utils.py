import sys
import select
from typing import Optional
from enum import Enum

def key_available():
    rlist, _, _ = select.select([sys.stdin], [], [], 1)
    return bool(rlist)

class PressedKey(Enum):
    ArrowUp = "ArrowUp"
    ArrowDown = "ArrowDown"
    ArrowLeft = "ArrowLeft"
    ArrowRight = "ArrowRight"
    Enter = "Enter"
    Escape = "Escape"
    Delete = "Delete"

def read_key() -> Optional[PressedKey]:
    ch = sys.stdin.read(1)
    if ch == "\n":
        return PressedKey.Enter
    if ch == "\x1b":
        if key_available():    
            key = sys.stdin.read(2)
            if key == "[A":
                return PressedKey.ArrowUp
            elif key == "[B":
                return PressedKey.ArrowDown
            elif key == "[D":
                return PressedKey.ArrowLeft
            elif key == "[C":
                return PressedKey.ArrowRight
            elif key == "[3":
                return PressedKey.Delete
        else:
            return PressedKey.Escape
