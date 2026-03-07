import sys
import os
import select
from typing import Optional
from enum import Enum

def key_available(timeout: float = 0.0):
    rlist, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
    return bool(rlist)

class PressedKey(Enum):
    ArrowUp = "ArrowUp"
    ArrowDown = "ArrowDown"
    ArrowLeft = "ArrowLeft"
    ArrowRight = "ArrowRight"
    Enter = "Enter"
    Escape = "Escape"
    Delete = "Delete"


def _read_byte() -> str:
    data = os.read(sys.stdin.fileno(), 1)
    return data.decode("utf-8", errors="ignore") if data else ""

def read_key() -> Optional[PressedKey]:
    ch = _read_byte()
    if ch == "\n" or ch == "\r":
        return PressedKey.Enter
    if ch == "\x1b":
        # Distinguish bare Escape from escape sequences (arrows/delete/etc.)
        # by waiting briefly for continuation bytes.
        if not key_available(0.02):
            return PressedKey.Escape

        seq = ""
        # Read a short sequence like "[A" or "[3~".
        while key_available(0.001) and len(seq) < 8:
            seq += _read_byte()
            # Common final bytes for CSI key sequences.
            if seq and (seq[-1].isalpha() or seq[-1] == "~"):
                break

        if seq == "[A":
            return PressedKey.ArrowUp
        if seq == "[B":
            return PressedKey.ArrowDown
        if seq == "[D":
            return PressedKey.ArrowLeft
        if seq == "[C":
            return PressedKey.ArrowRight
        if seq == "[3~":
            return PressedKey.Delete

        # Fallback: unknown escape sequence acts as Escape.
        return PressedKey.Escape
