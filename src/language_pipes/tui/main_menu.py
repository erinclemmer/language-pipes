import os
import toml
from time import sleep
from pathlib import Path
from typing import Tuple, Optional, List

from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import key_available, read_key, PressedKey
from language_pipes.tui.prompt import prompt, select_option
from language_pipes.util.config import get_config_files, default_config_dir

def load_libraries(window: TuiWindow):
    pipe_id = window.add_text(TermText(""), (20, 10))
    loading_id = window.add_text(TermText(""), (20, 12))
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

def new_config(window: TuiWindow, left_bound: int) -> Optional[str]:
    return prompt(TermText("Configuration Name"), window, (left_bound + 2, 10))

def prompt_node_id(window: TuiWindow, left_bound: int):
    return prompt(TermText("Node ID"), window, (left_bound, 10))

def select_node_id(window: TuiWindow, left_bound: int, credentials: List[str]):
    cred_text_ids = []
    for i, c in enumerate(credentials):
        cred_text_ids.append(window.add_text(TermText(c), (left_bound, 6 + (i * 2))))

def handle_file_load(window: TuiWindow, left_bound: int, config_file: Path):
    window.remove_all()
    window.paint()
    with open(config_file, 'r') as f:
        data = toml.load(f)
    cred_dir = default_config_dir() + "/" + "credentials"
    node_id = data.get("node_id", None)
    if len(os.listdir(cred_dir)) == 0:
        node_id = prompt_node_id(window, left_bound)
        if node_id is None:
            return None

def main_menu(termsize: Tuple[int, int]):
    with open('src/language_pipes/tui/banner.txt', 'r') as f:
        banner_text = f.read()

    left_bound = int((termsize[0] / 2.0) - 40.0)
    window = TuiWindow(termsize, (left_bound, 0))
    window.add_text(TermText(banner_text), (0, 0))
    window.add_text(TermText("Version X.X.X"), (0, 7))
    window.paint()
    load_libraries(window)

    main_menu_options = ["New Configuration", "Exit"]
    if len(get_config_files(default_config_dir() + "/configs")) > 0:
        main_menu_options.append("Load Configuration")

    res = select_option((left_bound + 10, 10), main_menu_options)
    if res is None or res == "Exit":
        exit()

