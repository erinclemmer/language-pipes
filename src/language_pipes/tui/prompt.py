import sys

from typing import Tuple, Optional, List
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import read_key, PressedKey
from language_pipes.tui.screen_utils import move_cursor

def prompt(txt: TermText, window: TuiWindow, pos: Tuple[int, int]) -> Optional[str]:
    txt.value += "|> "
    label_id = window.add_text(txt, pos)
    window.paint()
    start_idx = pos[0] + len(txt.value)
    cursor_idx = start_idx
    buffer_id = window.add_text(TermText(""), (cursor_idx, pos[1]))
    buffer = window.get_text(buffer_id)
    move_cursor(pos[1], window.position[0] + cursor_idx)

    def done():
        window.remove_txt(label_id)
        window.remove_txt(buffer_id)
        window.paint()

    while True:
        ch = sys.stdin.read(1)
        if ch.isnumeric() or ch.isalpha() or ch == "_" or ch == "-":
            window.update_text(buffer_id, TermText(buffer.text.value + ch))
            cursor_idx += 1
            window.paint()
            move_cursor(pos[1], window.position[0] + cursor_idx)
        elif ch == "\x7f" and cursor_idx > start_idx: # Backspace
            cursor_idx -= 1
            window.update_text(buffer_id, TermText(buffer.text.value[:-1]))
            window.paint()
            move_cursor(pos[1], window.position[0] + cursor_idx)
        elif ch == "\n" or ch == "\r": # Accept input [Enter]
            res = buffer.text.value
            done()
            return res
        elif ch == "\x1b": # Escape
            done()
            return None

def select_option( 
        pos: Tuple[int, int],
        options: List[str],
        msg: Optional[TermText] = None,
        allow_delete: bool = False
    ) -> Optional[Tuple[str, int]]:
    max_len = max([len(o) for o in options])
    help_text = "[Arrows]: Navigate, [Enter]: Accept, [Esc]: Back"
    if allow_delete:
        help_text += ", [Delete] Delete"
    width = max(max_len + 6, len(help_text), len(msg.value) if msg is not None else 0)
    window = TuiWindow((width, len(options) * 2 + 5), pos)

    mid_point = width / 2
    option_ids = []

    top_bound = 0
    if msg is not None:
        window.add_text(msg, (int(mid_point - len(msg.value) / 2), 0))
        top_bound += 2

    for i, opt in enumerate(options):
        l_bound = mid_point - (len(opt) / 2)
        option_ids.append(window.add_text(TermText(opt), (int(l_bound), top_bound + (i * 2))))

    window.add_text(TermText(help_text), (
        0, 
        top_bound + (len(options) * 2) + 2
    ))

    first_opt = window.get_text(option_ids[0])
    l_cursor_id = window.add_text(TermText("|>"), (first_opt.position[0] - 3, top_bound))
    r_cursor_id = window.add_text(TermText("<|"), (first_opt.position[0] + len(first_opt.text.value) + 1, top_bound))
    
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
            return options[selection_idx], 0

        if key == PressedKey.Delete and allow_delete:
            window.remove_all()
            window.paint()
            return options[selection_idx], 1

        if update:
            opt = window.get_text(option_ids[selection_idx])
            level = selection_idx * 2
            window.update_text(l_cursor_id, None, (
                opt.position[0] - 3,
                top_bound + level
            ))
            window.update_text(r_cursor_id, None, (
                opt.position[0] + len(opt.text.value) + 1,
                top_bound + level
            ))
            window.paint()

def prompt_bool(msg: TermText, pos: Tuple[int, int]) -> Optional[bool]:
    res = select_option(
        pos, ["Yes", "No"], msg
    )
    if res is None:
        return None
    return res[0] == "Yes"
