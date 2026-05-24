import os
from time import sleep
from pathlib import Path
from threading import Thread
from typing import Tuple, Optional

from ansinout import TuiWindow, TermText
from language_pipes.tui.util.prompt import prompt, select_option, prompt_bool
from language_pipes.tui.util.text import make_footer_text
from language_pipes.util.config import (
    get_config_files,
    get_app_dir,
    get_model_dir
)
from language_pipes.cli import VERSION

libraries_loaded = False


def load_libraries(window: TuiWindow):
    global libraries_loaded
    if libraries_loaded:
        return
    libraries_loaded = True
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
    
    paint_loader(10)
    window.update_text(loading_id, TermText("Loading: torch"))
    window.paint()
    import torch  # noqa: F401
    
    paint_loader(20)
    window.update_text(loading_id, TermText("Loading: transformers"))
    window.paint()
    import transformers  # noqa: F401

    paint_loader(30)
    window.update_text(loading_id, TermText("Loading: language-pipes"))
    window.paint()
    import language_pipes.content_provider.content_provider
    import language_pipes.content_provider.network_provider
    import language_pipes.content_provider.model_provider
    import language_pipes.tui.frame.main_frame  # noqa: F401

    paint_loader(40)
    window.update_text(loading_id, TermText("Loading Complete"))
    window.paint()
    sleep(0.2)
    window.remove_txt(pipe_id)
    window.remove_txt(loading_id)
    window.paint()


def new_config(window: TuiWindow) -> Optional[str]:
    footer_id = window.add_text(TermText(make_footer_text(["[A-Z]: Type", "Enter: Accept name", "Esc: Discard"])), (2, 23))
    res = prompt(TermText("Configuration Name"), window, (2, 11))
    window.remove_txt(footer_id)
    return res

def handle_file_load(
    window: TuiWindow, left_bound: int, termsize: Tuple[int, int], config_file: Path, auto_start: bool
):
    window.remove_all()
    window.paint()

    from language_pipes.tui.frame.main_frame import MainFrame
    frame = MainFrame((80, termsize[1]), (left_bound, 0), config_file=config_file, auto_start=auto_start)
    action = frame.run()
    if action == "exit":
        return "exit"
    return None


BANNER = r"""
 _                                                   ____   _
| |                                                 |  __`\(_)                
| |     __ _  ___   ___  _   _  __ _  __ _  ___     | |__) | |_ __   ___  ___ 
| |    / _` |/ _ \ / _ `| | | |/ _` |/ _` |/ _ \    |  ___/| | '_ \ / _ \/ __|
| |___| (_| | | | | (_| | |_| | (_| | (_| |  __/    | |    | | |_) |  __/\__ \
|______\__,_|_| |_|\__, |\__,_|\__,_|\__, |\___|    |_|    |_| .__/ \___||___/
                    __/ |             __/ |                  | |              
                   |___/             |___/                   |_|      
"""


def main_menu(termsize: Tuple[int, int], config_file: Optional[str], auto_start: bool):
    if config_file is None:
        auto_start = False

    app_dir = get_app_dir()
    model_dir = get_model_dir()

    if not os.path.exists(app_dir):
        app_dir.mkdir(parents=True)

    if not os.path.exists(model_dir):
        model_dir.mkdir(parents=True)

    config_dir = app_dir / "configs"
    if not os.path.exists(config_dir):
        config_dir.mkdir(parents=True)

    cred_dir = app_dir / "credentials"
    if not os.path.exists(cred_dir):
        cred_dir.mkdir(parents=True)

    log_dir = app_dir / "logs"
    if not os.path.exists(log_dir):
        log_dir.mkdir(parents=True)

    left_bound = int((termsize[0] / 2.0) - 40.0)
    window = TuiWindow((80, termsize[1]), (left_bound, 0))
    window.add_text(TermText(BANNER), (0, 0))
    window.add_text(TermText(f"Version {VERSION}"), (0, 8))
    window.paint()
    load_libraries(window)

    def restart():
        window.remove_all()
        window.paint()
        t = Thread(target=main_menu, args=(termsize, None, False, ))
        t.start()
        t.join()

    if config_file is not None:
        # Because the window has been initialized we need to assume that the config file is a valid path
        config_path: Optional[Path] = None
        if ".toml" in config_file:
            config_path = Path(config_file)
        else:
            config_path = config_dir / (config_file + ".toml")
        
        res = handle_file_load(window, left_bound, termsize, config_path, auto_start)
        if res == "exit":
            exit()
        if res is None:
            restart()
            return

    main_menu_options = ["New Configuration"]
    if len(get_config_files(config_dir)) > 0:
        main_menu_options.append("Load Configuration")

    main_menu_options.append("Exit")

    help_height = min(termsize[1] - 2, 23) - 10

    res = select_option((left_bound, 10), help_height, main_menu_options)
    if res is None or res == "Exit":
        exit()

    cmd = res[0]

    if cmd == "New Configuration":
        new_config_file = new_config(window)
        if new_config_file is None or new_config_file == "":
            restart()
            return
        
        config_path = config_dir / (new_config_file + ".toml")
        config_path.touch()
        res = handle_file_load(window, left_bound, termsize, config_path, auto_start)
        if res == "exit":
            exit()
        if res is None:
            return restart()

    if cmd == "Load Configuration":
        configs = get_config_files(config_dir)
        res = select_option(
            (left_bound, 10), help_height, configs, TermText("Select Configuration"), True
        )
        if res is None:
            return restart()
        config_file, cmd = res
        config_path = config_dir / (config_file + ".toml")
        if cmd == 0:
            res = handle_file_load(window, left_bound, termsize, config_path, auto_start)
            if res == "exit":
                exit()
            if res is None:
                return restart()
        if cmd == 1:
            res = prompt_bool(TermText(f"Delete {config_file}?"), (left_bound, 10), help_height)
            if res:
                os.remove(str(config_path))
            return restart()
