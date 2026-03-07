import os
import toml
from threading import Thread
from time import sleep
from pathlib import Path
from typing import Tuple, Optional, List

from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import key_available, read_key, PressedKey
from language_pipes.tui.prompt import prompt, select_option, prompt_bool
from language_pipes.util.config import get_config_files, default_config_dir, default_model_dir

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

def new_config(window: TuiWindow) -> Optional[str]:
    return prompt(TermText("Configuration Name"), window, (2, 10))

def prompt_node_id(window: TuiWindow):
    cred_dir = Path(default_config_dir(), "credentials")
    credentials = os.listdir(cred_dir)
    if len(credentials) == 0:
        node_id = prompt(TermText("Node ID"), window, (2, 10))
        if node_id is None or node_id == '':
            return None
        return node_id
    res = select_option((0, 0), credentials)

def select_node_id(window: TuiWindow, left_bound: int, credentials: List[str]):
    cred_text_ids = []
    for i, c in enumerate(credentials):
        cred_text_ids.append(window.add_text(TermText(c), (left_bound, 6 + (i * 2))))

def handle_file_load(window: TuiWindow, config_file: Path):
    with open(config_file, 'r') as f:
        data = toml.load(f)
    node_id = data.get("node_id", None)
    if node_id is None:
        data["node_id"] = prompt_node_id(window)
        with open(config_file, 'w') as f:
            toml.dump(data, f)
    

def main_menu(termsize: Tuple[int, int]):
    with open('src/language_pipes/tui/banner.txt', 'r') as f:
        banner_text = f.read()

    app_dir = default_config_dir()
    model_dir = default_model_dir()
    if not os.path.exists(app_dir):
        Path(app_dir).mkdir(parents=True)
    
    config_dir = str(Path(app_dir) / "configs")
    if not os.path.exists(config_dir):
        Path(config_dir).mkdir(parents=True)

    cred_dir = str(Path(app_dir) / "credentials")
    if not os.path.exists(cred_dir):
        Path(cred_dir).mkdir(parents=True)

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True)

    left_bound = int((termsize[0] / 2.0) - 40.0)
    window = TuiWindow((80, termsize[1]), (left_bound, 0))
    window.add_text(TermText(banner_text), (0, 0))
    window.add_text(TermText("Version X.X.X"), (0, 7))
    window.paint()
    load_libraries(window)

    main_menu_options = ["New Configuration"]
    if len(get_config_files(default_config_dir() + "/configs")) > 0:
        main_menu_options.append("Load Configuration")
    
    main_menu_options.append("Exit")

    res = select_option((left_bound + 10, 10), main_menu_options)
    if res is None or res == "Exit":
        exit()

    cmd = res[0]

    def restart():
        window.remove_all()
        window.paint()
        t = Thread(target=main_menu, args=(termsize, ))
        t.start()
        t.join()

    if cmd == "New Configuration":
        config_file = new_config(window)
        if config_file is None or config_file == '':
            restart()
            return
        config_path = Path(default_config_dir(), "configs", config_file + ".toml")
        config_path.touch()
        res = handle_file_load(window, config_path)
        if res is None:
            return restart()
        
    if cmd == "Load Configuration":
        window.remove_all()
        window.paint()
        configs = get_config_files(str(Path(default_config_dir(), "configs")))
        res = select_option((left_bound + 2, 0), configs, TermText("Select Configuration"), True)
        if res is None:
            return restart()
        config_file, cmd = res
        config_path = Path(default_config_dir(), "configs", config_file + ".toml")
        if cmd == 0:
            res = handle_file_load(window, config_path)
            if res is None:
                return restart()
        if cmd == 1:
            res = prompt_bool(TermText(f"Delete {config_file}?"), (left_bound + 2, 9))
            if res:
                os.remove(str(config_path))
            return restart()