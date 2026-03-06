import sys

from typing import Tuple, Optional, List
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import read_key, PressedKey
from language_pipes.tui.screen_utils import print_pos, move_cursor

def prompt(txt: TermText, window: TuiWindow, pos: Tuple[int, int]) -> Optional[str]:
    txt.value += "|> "
    label_id = window.add_text(txt, pos)
    window.paint()
    start_idx = pos[0] + len(txt.value)
    cursor_idx = start_idx
    buffer = ""
    print_pos(pos[1], cursor_idx, '')

    while True:
        ch = sys.stdin.read(1)
        if ch.isnumeric() or ch.isalpha() or ch == "_" or ch == "-":
            print_pos(pos[1], cursor_idx, ch)
            buffer += ch
            cursor_idx += 1
        elif ch == "\x7f" and cursor_idx > start_idx: # Backspace
            cursor_idx -= 1
            buffer[:-1]
            print_pos(pos[1], cursor_idx, ' ')
            move_cursor(pos[1], cursor_idx)
        elif ch == "\n": # Accept input [Enter]
            return buffer
        elif ch == "\x1b": # Escape
            window.add_text(TermText(" "  * len(buffer)), (start_idx, pos[1]))
            window.remove_txt(label_id)
            window.paint()
            return None

def select_option( 
        pos: Tuple[int, int],
        options: List[str]
    ) -> Optional[str]:
    max_len = max([len(o) for o in options])
    help_text = "[Arrow Keys]: Navigate, [Enter]: Accept Selection"
    width = max(max_len + 6, len(help_text))
    window = TuiWindow((width, len(options) * 2 + 3), pos)

    mid_point = width / 2
    option_ids = []

    for i, opt in enumerate(options):
        l_bound = mid_point - (len(opt) / 2)
        option_ids.append(window.add_text(TermText(opt), (int(l_bound), (i * 2))))

    window.add_text(TermText(help_text), (
        0, 
        (len(options) * 2) + 2
    ))

    first_opt = window.get_text(option_ids[0])
    l_cursor_id = window.add_text(TermText("|>"), (first_opt.position[0] - 3, 0))
    r_cursor_id = window.add_text(TermText("<|"), (first_opt.position[0] + len(first_opt.text.value) + 1, 0))
    
    window.paint()
    selection_idx = 0
    while True:
        key = read_key()
        update = False
    
        if key == PressedKey.ArrowUp:
            update = True
            selection_idx = selection_idx - 1
            if selection_idx == -1:
                selection_idx = len(options) - 1
        if key == PressedKey.ArrowDown:
            update = True
            selection_idx = selection_idx + 1
            if selection_idx == len(options):
                selection_idx = 0
        
        if key == PressedKey.Escape:
            window.remove_all()
            window.paint()
            return None

        if key == PressedKey.Enter:
            window.remove_all()
            window.paint()
            return options[selection_idx]

        if update:
            opt = window.get_text(option_ids[selection_idx])
            level = selection_idx * 2
            window.update_text(l_cursor_id, None, (
                opt.position[0] - 3,
                level
            ))
            window.update_text(r_cursor_id, None, (
                opt.position[0] + len(opt.text.value) + 1,
                level
            ))
            window.paint()


