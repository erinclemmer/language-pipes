import os
import toml
import shutil
from threading import Thread
from time import sleep
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.distributed_state_network.util.key_manager import CredentialManager
from language_pipes.tui.prompt import prompt, select_option, prompt_bool
from language_pipes.util.config import get_config_files, default_config_dir, default_model_dir

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

def prompt_node_id(window: TuiWindow, left_bound: int):
    window.remove_all()
    window.paint()
    cred_dir = Path(default_config_dir(), "credentials")
    credentials = os.listdir(cred_dir)
    if len(credentials) == 0:
        node_id = prompt(TermText("Node ID"), window, (2, 0))
        if node_id is None or node_id == '':
            return None
        cred_manager = CredentialManager(str(cred_dir), node_id)
        cred_manager.generate_keys()
        return node_id
    credentials.append("New Node ID")
    res = select_option((left_bound, 0), credentials, TermText("Select Node ID"), True)
    if res is None:
        return None
    cred, cmd = res
    if cmd == 1:
        if cred == "New Node ID":
            return None
        res = prompt_bool(TermText(f"Delete {cred} credentials"), (left_bound, 0))
        if res is None:
            return None
        if res:
            shutil.rmtree(str(cred_dir / cred))
        return None
    
    if cred == "New Node ID":
        node_id = prompt(TermText("Node ID"), window, (left_bound, 0))
        if node_id is None or node_id == '':
            return None
        cred_manager = CredentialManager(str(cred_dir), node_id)
        cred_manager.generate_keys()
        return node_id
    else:
        return cred

def select_node_id(window: TuiWindow, left_bound: int, credentials: List[str]):
    cred_text_ids = []
    for i, c in enumerate(credentials):
        cred_text_ids.append(window.add_text(TermText(c), (left_bound, 6 + (i * 2))))

def get_bootstrap_node(window: TuiWindow, pos: Tuple[int, int]) -> Optional[Tuple[str, int]]:
    res = prompt(TermText("Connect to IP Address"), window, pos)
    if res is None:
        return None
    
    bootstrap_ip = res
    connect_id = window.add_text(TermText(f"Connect to: {res}"), pos)
    window.paint()
    res = prompt(TermText("Connect to Port"), window, (pos[0], pos[1] + 1))
    if res is None:
        return None
    
    bootstrap_port = res
    try:
        bootstrap_port = int(bootstrap_port)
    except ValueError:
        return None
    window.remove_txt(connect_id)
    return bootstrap_ip, bootstrap_port

def handle_join_network(window: TuiWindow, create: bool) -> Optional[Dict]:
    current_step = 0
    window.add_text(TermText("Create Network" if create else "Join Network"), (10, 0))
    username_id = window.add_text(TermText("         username|>"), (0, 2))
    password_id = window.add_text(TermText("         password|>"), (0, 3))
    bootstrap_address_id = None
    bootstrap_port_id = None
    if not create:
        bootstrap_address_id = window.add_text(TermText("bootstrap address|>"), (0, 4))
        bootstrap_port_id = window.add_text(TermText("   bootstrap port|>"), (0, 5))
    window.paint()
    username_buf = ""
    password_buf = ""
    bootstrap_addr_buf = ""
    bootstrap_port_buf = ""

    def get_pwd_str():
        return "".join(["#" for _ in range(0, len(password_buf))])

    while True:
        if current_step == 0:
            window.hide_txt(username_id)
            window.paint()
            res = prompt(TermText("         username"), window, (0, 2), username_buf)
            if res is None:
                return None
            username_buf = res
            current_step += 1
            window.show_txt(username_id)
            window.update_text(username_id, TermText(f"         username|> {username_buf}"))
        elif current_step == 1:
            window.hide_txt(password_id)
            window.paint()
            res = prompt(TermText("         password"), window, (0, 3), password_buf)
            if res is None:
                current_step -= 1
                window.show_txt(password_id)
                window.update_text(password_id, TermText(f"         password|> {get_pwd_str()}"))
                continue
            password_buf = res
            current_step += 1
            window.show_txt(password_id)
            window.update_text(password_id, TermText(f"         password|> {get_pwd_str()}"))
            if create:
                return {
                    "username": username_buf,
                    "password": password_buf,
                    "bootstrap_addr": None,
                    "bootstrap_port": None
                }
        elif current_step == 2 and bootstrap_address_id is not None:
            window.hide_txt(bootstrap_address_id)
            window.paint()
            res = prompt(TermText("bootstrap address"), window, (0, 4), bootstrap_addr_buf)
            if res is None:
                current_step -= 1
                window.show_txt(bootstrap_address_id)
                window.update_text(bootstrap_address_id, TermText(f"bootstrap address|> {bootstrap_addr_buf}"))
                continue
            bootstrap_addr_buf = res
            current_step += 1
            window.show_txt(bootstrap_address_id)
            window.update_text(bootstrap_address_id, TermText(f"bootstrap address|> {bootstrap_addr_buf}"))
        elif current_step == 3 and bootstrap_port_id is not None:
            window.hide_txt(bootstrap_port_id)
            window.paint()
            res = prompt(TermText("   bootstrap port"), window, (0, 5), bootstrap_port_buf)
            if res is None:
                current_step -= 1
                window.show_txt(bootstrap_port_id)
                window.update_text(bootstrap_port_id, TermText(f"   bootstrap port|> {bootstrap_port_buf}"))
                continue
            bootstrap_port_buf = res
            window.remove_all()
            window.paint()
            return {
                "username": username_buf,
                "password": password_buf,
                "bootstrap_addr": bootstrap_addr_buf,
                "bootstrap_port": bootstrap_port_buf
            }

def handle_file_load(window: TuiWindow, left_bound: int, config_file: Path):
    window.remove_all()
    window.paint()
    with open(config_file, 'r') as f:
        data = toml.load(f)
    node_id = data.get("node_id", None)

    def save_data(d):
        with open(config_file, 'w') as f:
            toml.dump(d, f)

    if node_id is None:
        res = select_option((left_bound, 0), ["Create Network", "Join Network"], TermText("Select an option"))
        if res is None:
            return
        cmd, _ = res
        res = handle_join_network(window, cmd == "Create Network")
        if res is None:
            return
        data["node_id"] = res["username"]
        data["aes_key"] = res["password"]
        if res["bootstrap_addr"] is not None:
            data["bootstrap_nodes"] = [{
                "address": res["bootstrap_addr"],
                "port": res["bootstrap_port"]
            }]
        save_data(data)

    prompt(TermText("TEST"), window, (0, 10))

    # aes_key = data.get("aes_key", None)
    # if aes_key is None:
            

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
        res = handle_file_load(window, left_bound, config_path)
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
            res = handle_file_load(window, left_bound, config_path)
            if res is None:
                return restart()
        if cmd == 1:
            res = prompt_bool(TermText(f"Delete {config_file}?"), (left_bound + 2, 0))
            if res:
                os.remove(str(config_path))
            return restart()