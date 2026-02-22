import sys
from time import sleep
from typing import Tuple, Optional
from tui import TuiWindow, TermText
from kb_utils import key_available, read_key, PressedKey
from screen_utils import print_pos, move_cursor

def prompt(txt: TermText, window: TuiWindow, pos: Tuple[int, int]) -> Optional[str]:
    txt.value += "|> "
    label_id = window.add_text(txt, pos)
    window.paint()
    start_idx = pos[0] + len(txt.value)
    cursor_idx = start_idx
    buffer = ""
    print_pos(pos[1], cursor_idx, '')
    while True:
        if key_available():
            ch = sys.stdin.read(1)
            if ch.isalpha():
                print_pos(pos[1], cursor_idx, ch)
                buffer += ch
                cursor_idx += 1
            if ch == "\x7f" and cursor_idx > start_idx:
                cursor_idx -= 1
                buffer[:-1]
                print_pos(pos[1], cursor_idx, ' ')
                move_cursor(pos[1], cursor_idx)
            if ch == "\n":
                return buffer
            if ch == "\x1b":
                window.add_text(TermText(" "  * len(buffer)), (start_idx, pos[1]))
                window.remove_txt(label_id)
                window.paint()
                return None
        sleep(0.1)