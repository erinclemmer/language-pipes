import sys
import termios
import tty
from enum import Enum
from typing import Optional

ESC = "\x1b["
ALT_SCR_ENTER = f"{ESC}?1049h"
ALT_SCR_EXIT  = f"{ESC}?1049l"
CLS = f"{ESC}2J"
HOME = f"{ESC}H"

def write(s: str):
    sys.stdout.write(s)
    sys.stdout.flush()

def enable_vt_mode():
    """
    Linux terminals typically already support ANSI output.
    Put stdin into cbreak mode so keypresses are immediate (no Enter),
    and disable echo so keys are not printed.
    """
    fd_in = sys.stdin.fileno()
    old_in_attrs = termios.tcgetattr(fd_in)
    tty.setcbreak(fd_in)
    write(ALT_SCR_ENTER + CLS + HOME)
    return (fd_in, old_in_attrs)

def exit_vt_mode():
    write(ALT_SCR_EXIT)

def restore_mode(fd_in, old_in_attrs):
    termios.tcsetattr(fd_in, termios.TCSADRAIN, old_in_attrs)

def print_pos(row: int, col: int, s: str, fg: Optional['Color'] = None, bg: Optional['BgColor'] = None, bold: bool = False):
    # ANSI positions are 1-based
    s = color(s, fg, bg, bold)
    write(f"{ESC}{row + 1};{col + 1}H{s}")

def move_cursor(row: int, col: int):
    write(f"{ESC}{row + 1};{col + 1}H")

class CursorTypes(Enum):
    Default = 1
    Blinking_Block = 1
    Steady_Block = 2
    Blinking_Underline = 3
    Steady_Underline = 4
    Blinking_Bar = 5
    Steady_Bar = 6

def change_cursor(t: CursorTypes):
    write(f"\033[{t.value} q")

class Color(Enum):
    # Standard foreground colors (30-37)
    Black = 30
    Red = 31
    Green = 32
    Yellow = 33
    Blue = 34
    Magenta = 35
    Cyan = 36
    White = 37
    Default = 39
    # Bright foreground colors (90-97)
    BrightBlack = 90
    Gray = 90
    BrightRed = 91
    BrightGreen = 92
    BrightYellow = 93
    BrightBlue = 94
    BrightMagenta = 95
    BrightCyan = 96
    BrightWhite = 97


class BgColor(Enum):
    # Standard background colors (40-47)
    Black = 40
    Red = 41
    Green = 42
    Yellow = 43
    Blue = 44
    Magenta = 45
    Cyan = 46
    White = 47
    Default = 49
    # Bright background colors (100-107)
    BrightBlack = 100
    Gray = 100
    BrightRed = 101
    BrightGreen = 102
    BrightYellow = 103
    BrightBlue = 104
    BrightMagenta = 105
    BrightCyan = 106
    BrightWhite = 107


def color(text: str, fg: Optional[Color] = None, bg: Optional[BgColor] = None, bold: bool = False) -> str:
    codes = []
    if bold:
        codes.append("1")

    if isinstance(fg, Color):
        codes.append(str(fg.value))

    if isinstance(bg, BgColor):
        codes.append(str(bg.value))

    if len(codes) == 0:
        return text

    prefix = f"\033[{';'.join(codes)}m"
    reset = "\033[0m"
    return f"{prefix}{text}{reset}"
