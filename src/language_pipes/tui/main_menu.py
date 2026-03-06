from time import sleep
from typing import Tuple

from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import key_available, read_key, PressedKey
from language_pipes.tui.prompt import prompt
from language_pipes.util.config import get_config_files, default_config_dir

def load_libraries(window: TuiWindow, banner_x: int):
    pipe_id = window.add_text(TermText(""), (banner_x + 20, 10))
    loading_id = window.add_text(TermText(""), (banner_x + 20, 12))
    num_total_segments = 40
    def paint_loader(i):
        pipe_v: str = "|>" + ("=" * i) + (" " * (num_total_segments - i)) + "<|"
        window.update_text(pipe_id, TermText(pipe_v))
        window.paint()
    paint_loader(0)
    window.update_text(loading_id, TermText("Loading: inspect"))
    window.paint()
    import inspect
    paint_loader(10)
    window.update_text(loading_id, TermText("Loading: numpy"))
    window.paint()
    import numpy
    paint_loader(20)
    window.update_text(loading_id, TermText("Loading: torch"))
    window.paint()
    import torch
    paint_loader(30)
    window.update_text(loading_id, TermText("Loading: transformers"))
    window.paint()
    import transformers
    paint_loader(40)
    window.update_text(loading_id, TermText("Loading Complete"))
    window.paint()
    sleep(0.1)
    window.remove_txt(pipe_id)
    window.remove_txt(loading_id)
    window.paint()

def new_config(window: TuiWindow, left_bound: int) -> bool:
    res = prompt(TermText("Configuration Name"), window, (left_bound + 2, 10))
    if res is None:
        return False
    return True

def main_menu(window: TuiWindow, termsize: Tuple[int, int]):
    with open('src/language_pipes/tui/banner.txt', 'r') as f:
        banner_text = f.read()

    left_bound = int((termsize[0] / 2.0) - 40.0)
    window.add_text(TermText(banner_text), (left_bound, 0))
    window.add_text(TermText("Version X.X.X"), (left_bound, 7))
    load_libraries(window, left_bound)

    new_config_txt = TermText("New Configuration")

    has_config_files = len(get_config_files(default_config_dir() + "/configs")) > 0
    load_config_txt = TermText("Load Configuration") if has_config_files else None
    
    def build_options():
        l_cursor_id = window.add_text(TermText("|>"), (left_bound + 20, 10))
        r_cursor_id = window.add_text(TermText("<|"), (left_bound + 42, 10))
        new_config_id = window.add_text(new_config_txt, (left_bound + 23, 10))
        load_config_id = None
        if load_config_txt is not None:
            load_config_id = window.add_text(load_config_txt, (left_bound + 23, 12))
        return l_cursor_id, r_cursor_id, new_config_id, load_config_id
    
    l_cursor_id, r_cursor_id, new_config_id, load_config_id = build_options()
    
    window.paint()
    cursor = False
    while True:
        if key_available():
            key = read_key()
            if key is None:
                continue
            if key == PressedKey.ArrowUp or key == PressedKey.ArrowDown:
                cursor = not cursor
            if key == PressedKey.Enter:
                if not cursor:
                    window.remove_txt(l_cursor_id)
                    window.remove_txt(r_cursor_id)
                    window.remove_txt(new_config_id)
                    if load_config_id is not None:
                        window.remove_txt(load_config_id)
                    if not new_config(window, left_bound):
                        l_cursor_id, r_cursor_id, new_config_id, load_config_id = build_options()
                    else:
                        return
            window.update_text(l_cursor_id, v=None, pos=(left_bound + 20, 12 if cursor else 10))
            window.update_text(r_cursor_id, v=None, pos=(left_bound + 42, 12 if cursor else 10))
            window.paint()
        sleep(0.02)
