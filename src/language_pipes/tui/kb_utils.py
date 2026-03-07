import sys
import os
import select
from typing import Optional, Tuple
from enum import Enum

def key_available(timeout: float = 0.0):
    rlist, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
    return bool(rlist)

class PressedKey(Enum):
    Alpha = "Alpha"
    ArrowUp = "ArrowUp"
    ArrowDown = "ArrowDown"
    ArrowLeft = "ArrowLeft"
    ArrowRight = "ArrowRight"
    Backspace = "Backspace"
    Enter = "Enter"
    Escape = "Escape"
    Delete = "Delete"
    Nop = "Nop"

def _read_byte() -> str:
    data = os.read(sys.stdin.fileno(), 1)
    return data.decode("utf-8", errors="ignore") if data else ""

def read_key() -> Tuple[PressedKey, str]:
    accepted_chars = ["_", "-", "."]
    ch = _read_byte()
    if ch.isalpha() or ch.isnumeric() or ch in accepted_chars:
        return PressedKey.Alpha, ch
    if ch == "\n" or ch == "\r":
        return PressedKey.Enter, ch
    if ch == "\x7f":
        return PressedKey.Backspace, ch
    if ch == "\x1b":
        # Distinguish bare Escape from escape sequences (arrows/delete/etc.)
        # by waiting briefly for continuation bytes.
        if not key_available(0.02):
            return PressedKey.Escape, ch

        seq = ""
        # Read a short sequence like "[A" or "[3~".
        while key_available(0.001) and len(seq) < 8:
            seq += _read_byte()
            # Common final bytes for CSI key sequences.
            if seq and (seq[-1].isalpha() or seq[-1] == "~"):
                break

        if seq == "[A":
            return PressedKey.ArrowUp, seq
        if seq == "[B":
            return PressedKey.ArrowDown, seq
        if seq == "[D":
            return PressedKey.ArrowLeft, seq
        if seq == "[C":
            return PressedKey.ArrowRight, seq
        if seq == "[3~":
            return PressedKey.Delete, seq

        # Fallback: unknown escape sequence acts as Escape.
        return PressedKey.Escape, seq
    return PressedKey.Nop, ""