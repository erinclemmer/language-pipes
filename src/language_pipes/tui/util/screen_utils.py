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

def print_pos(row: int, col: int, s: str, fg: Optional[int] = None, bg: Optional[int] = None, bold: bool =False):
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

def color(text: str, fg=None, bg=None, bold=False) -> str:
    codes = []
    if bold:
        codes.append("1")

    # Foreground (4-bit SGR codes: 30-37, 90-97)
    if isinstance(fg, int):
        codes.append(str(fg))

    # Background (4-bit SGR codes: 40-47, 100-107)
    if isinstance(bg, int):
        codes.append(str(bg))

    if len(codes) == 0:
        return text

    prefix = f"\033[{';'.join(codes)}m"
    reset = "\033[0m"
    return f"{prefix}{text}{reset}"
