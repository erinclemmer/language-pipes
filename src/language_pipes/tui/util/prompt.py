from typing import Tuple, Optional, List
from ansinout import read_key, PressedKey, move_cursor, change_cursor, TuiWindow, TermText
from ansinout.screen import CursorTypes
from language_pipes.tui.util.text import make_footer_text

def prompt(txt: TermText, window: TuiWindow, pos: Tuple[int, int], initial: str = "") -> Optional[str]:
    change_cursor(CursorTypes.Blinking_Bar)
    txt.value += "|> "
    label_id = window.add_text(txt, pos)
    start_idx = pos[0] + len(txt.value)
    cursor_idx = len(initial)
    buffer_id = window.add_text(TermText(initial), (start_idx, pos[1]))
    buffer = window.get_text(buffer_id)
    assert buffer is not None
    window.paint()
    move_cursor(pos[1], window.position[0] + start_idx + cursor_idx)

    def done():
        window.remove_txt(label_id)
        window.remove_txt(buffer_id)
        window.paint()
        change_cursor(CursorTypes.Default)

    def update_cursor():
        move_cursor(pos[1], window.position[0] + start_idx + cursor_idx)

    while True:
        key, ch = read_key()
        if key == PressedKey.Alpha:
            window.update_text(buffer_id, TermText(buffer.text.value[:cursor_idx] + ch + buffer.text.value[cursor_idx:]))
            cursor_idx += 1
            window.paint()
            update_cursor()
        elif key == PressedKey.ArrowLeft:
            cursor_idx -= 1
            if cursor_idx < 0:
                cursor_idx = 0
            update_cursor()
        elif key == PressedKey.ArrowRight:
            cursor_idx += 1
            if cursor_idx > len(buffer.text.value):
                cursor_idx = len(buffer.text.value)
            update_cursor()
        elif key == PressedKey.Backspace and cursor_idx > 0: # Backspace
            cursor_idx -= 1
            window.update_text(buffer_id, TermText(buffer.text.value[:cursor_idx] + buffer.text.value[cursor_idx + 1:]))
            window.paint()
            update_cursor()
        elif key == PressedKey.Delete and cursor_idx < len(buffer.text.value):
            window.update_text(buffer_id, TermText(buffer.text.value[:cursor_idx] + buffer.text.value[cursor_idx + 1:]))
            window.paint()
            update_cursor()
        elif key == PressedKey.Enter: # Accept input [Enter]
            res = buffer.text.value
            done()
            return res
        elif key == PressedKey.Escape: # Escape
            done()
            return None

def select_option(
        pos: Tuple[int, int],
        height: int,
        options: List[str],
        msg: Optional[TermText] = None,
        allow_delete: bool = False
    ) -> Optional[Tuple[str, int]]:
    max_len = max([len(o) for o in options])
    help_text =  ""
    if allow_delete:
        help_text = make_footer_text(["Arrows U/D: Move", "Enter: Select", "Delete: Delete", "Esc: Back"])
    else:
        help_text = make_footer_text(["Arrows U/D: Move", "Enter: Select", "Esc: Back"])
    width = max(max_len + 6, len(help_text), len(msg.value) if msg is not None else 0)
    window = TuiWindow((width, 25), pos)

    mid_point = width / 2

    top_bound = 0
    if msg is not None:
        window.add_text(msg, (int(mid_point - len(msg.value) / 2), 0))
        top_bound = 3

    max_visible = max(1, (height - top_bound - 1) // 2)
    visible_count = min(max_visible, len(options))

    option_slots = []
    for j in range(visible_count):
        option_slots.append(window.add_text(TermText(""), (0, top_bound + j * 2)))

    window.add_text(TermText(help_text), (0, height))

    up_indicator_y = top_bound - 1 if top_bound > 0 else 0
    up_indicator_id = window.add_text(TermText(""), (int(mid_point), up_indicator_y))
    down_indicator_id = window.add_text(TermText(""), (int(mid_point), top_bound + visible_count * 2 - 1))

    l_cursor_id = window.add_text(TermText("|>"), (0, top_bound))
    r_cursor_id = window.add_text(TermText("<|"), (0, top_bound))

    selection_idx = 0
    scroll_offset = 0

    def render():
        nonlocal scroll_offset
        if selection_idx < scroll_offset:
            scroll_offset = selection_idx
        elif selection_idx >= scroll_offset + visible_count:
            scroll_offset = selection_idx - visible_count + 1

        for j, slot_id in enumerate(option_slots):
            option_i = scroll_offset + j
            if option_i < len(options):
                opt = options[option_i]
                l_bound = int(mid_point - (len(opt) / 2))
                window.update_text(slot_id, TermText(opt), (l_bound, top_bound + j * 2))
            else:
                window.update_text(slot_id, TermText(""), (0, top_bound + j * 2))

        window.update_text(up_indicator_id, TermText("^" if scroll_offset > 0 else ""))
        window.update_text(
            down_indicator_id,
            TermText("v" if scroll_offset + visible_count < len(options) else "")
        )

        visible_idx = selection_idx - scroll_offset
        opt_slot = window.get_text(option_slots[visible_idx])
        assert opt_slot is not None
        window.update_text(l_cursor_id, None, (
            opt_slot.position[0] - 3,
            top_bound + visible_idx * 2
        ))
        window.update_text(r_cursor_id, None, (
            opt_slot.position[0] + len(opt_slot.text.value) + 1,
            top_bound + visible_idx * 2
        ))
        window.paint()

    render()
    while True:
        key, _ = read_key()
        update = False

        if key == PressedKey.ArrowUp:
            update = True
            selection_idx = selection_idx - 1
            if selection_idx == -1:
                selection_idx = len(options) - 1
                scroll_offset = max(0, len(options) - visible_count)
        if key == PressedKey.ArrowDown:
            update = True
            selection_idx = selection_idx + 1
            if selection_idx == len(options):
                selection_idx = 0
                scroll_offset = 0

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
            render()

def prompt_bool(msg: TermText, pos: Tuple[int, int], height: int) -> Optional[bool]:
    res = select_option(
        pos, height, ["Yes", "No"], msg
    )
    if res is None:
        return None
    return res[0] == "Yes"
