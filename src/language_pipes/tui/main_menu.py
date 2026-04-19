import os
from time import sleep
from pathlib import Path
from threading import Thread
from typing import Tuple, Optional

from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.util.prompt import prompt, select_option, prompt_bool
from language_pipes.util.config import (
    get_config_files,
    get_app_dir,
    get_model_dir
)
from language_pipes.cli import VERSION
from language_pipes.tui.frame.provider_calls import ProviderCall

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
    import language_pipes.tui.content_provider.content_provider
    import language_pipes.tui.content_provider.network_provider
    import language_pipes.tui.content_provider.model_provider
    import language_pipes.tui.frame.main_frame  # noqa: F401

    paint_loader(40)
    window.update_text(loading_id, TermText("Loading Complete"))
    window.paint()
    sleep(0.2)
    window.remove_txt(pipe_id)
    window.remove_txt(loading_id)
    window.paint()


def new_config(window: TuiWindow) -> Optional[str]:
    return prompt(TermText("Configuration Name"), window, (2, 10))


def handle_file_load(
    window: TuiWindow, left_bound: int, termsize: Tuple[int, int], config_file: Path
):
    from language_pipes.tui.content_provider.content_provider import ContentProvider
    from language_pipes.tui.content_provider.network_provider import NetworkProvider
    from language_pipes.tui.content_provider.model_provider import ModelProvider

    window.remove_all()
    window.paint()
    
    content_provider = ContentProvider()
    
    providers = {
        ProviderCall.get_network_config: lambda: NetworkProvider.get_network_config(
            config_file
        ),
        ProviderCall.save_network_config: lambda data: (
            NetworkProvider.save_network_config(config_file, data)
        ),
        ProviderCall.get_registered_node_ids: NetworkProvider.get_registered_node_ids,
        ProviderCall.delete_node_id: NetworkProvider.delete_node_id,
        ProviderCall.save_new_node_id: NetworkProvider.save_new_node_id,
        ProviderCall.generate_aes_key: NetworkProvider.generate_aes_key,
        ProviderCall.validate_aes_key: NetworkProvider.validate_aes_key,
        ProviderCall.detect_network_ip: NetworkProvider.detect_network_ip,
        ProviderCall.start_network: lambda: (
            content_provider.network_provider.start_router(config_file)
        ),
        ProviderCall.stop_network: content_provider.network_provider.stop_router,
        ProviderCall.get_network_status: content_provider.network_provider.get_router_status,
        ProviderCall.get_total_system_ram: ContentProvider.get_total_system_ram,
        ProviderCall.get_used_system_ram: ContentProvider.get_used_system_ram,
        ProviderCall.list_peers: content_provider.network_provider.get_peers,
        ProviderCall.get_installed_models: ModelProvider.get_installed_models,
        ProviderCall.delete_installed_model: ModelProvider.delete_installed_model,
        ProviderCall.start_download: content_provider.model_provider.start_download,
        ProviderCall.stop_model_download: content_provider.model_provider.stop_model_download,
        ProviderCall.check_download_progress: content_provider.model_provider.check_download_progress,
        ProviderCall.get_hf_token: ModelProvider.get_hf_token,
        ProviderCall.save_hf_token: ModelProvider.save_hf_token,
        ProviderCall.get_model_manager_logs: content_provider.model_provider.get_model_manager_logs,
        ProviderCall.is_port_available: ContentProvider.is_port_available,

        ProviderCall.host_model: content_provider.model_provider.host_model,
        ProviderCall.get_models_to_load: lambda: ModelProvider.get_models_to_load(
            config_file
        ),
        ProviderCall.save_models_to_load: lambda m: ModelProvider.save_models_to_load(
            config_file, m
        ),
        ProviderCall.validate_device_name: ModelProvider.validate_device_name,
        ProviderCall.get_models_status: content_provider.model_provider.get_models_status,
        ProviderCall.shutdown_models: content_provider.model_provider.shutdown_models,

        
        ProviderCall.get_pipes_connected: content_provider.pipe_provider.get_connected_pipes,
        ProviderCall.get_network_pipes: content_provider.pipe_provider.get_network_pipes,

        
        ProviderCall.start_oai_server: content_provider.job_provider.start_oai_server,
        ProviderCall.stop_oai_server: content_provider.job_provider.stop_oai_server,
        ProviderCall.oai_server_running: content_provider.job_provider.oai_server_running,
        ProviderCall.get_oai_logs: content_provider.job_provider.get_oai_logs,
        ProviderCall.get_oai_port: lambda: content_provider.job_provider.get_oai_port(config_file),
        ProviderCall.set_oai_port: lambda p: content_provider.job_provider.set_oai_port(config_file, p),
        ProviderCall.get_api_keys: lambda: content_provider.job_provider.get_api_keys(config_file),
        ProviderCall.set_api_keys: lambda ks: content_provider.job_provider.set_api_keys(config_file, ks),
        ProviderCall.get_active_jobs: content_provider.job_provider.get_active_jobs,
        ProviderCall.shutdown: content_provider.shutdown
    }

    from language_pipes.tui.frame.main_frame import MainFrame
    frame = MainFrame((80, termsize[1]), (left_bound, 0), providers=providers)
    action = frame.run()
    if action == "exit":
        return "exit"
    return None

def main_menu(termsize: Tuple[int, int], config_file: Optional[str], auto_start: bool):
    with open("src/language_pipes/tui/banner.txt", "r") as f:
        banner_text = f.read()

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
    window.add_text(TermText(banner_text), (0, 0))
    window.add_text(TermText(f"Version {VERSION}"), (0, 7))
    window.paint()
    load_libraries(window)

    main_menu_options = ["New Configuration"]
    if len(get_config_files(config_dir)) > 0:
        main_menu_options.append("Load Configuration")

    main_menu_options.append("Exit")

    res = select_option((left_bound + 10, 10), main_menu_options)
    if res is None or res == "Exit":
        exit()

    cmd = res[0]

    def restart():
        window.remove_all()
        window.paint()
        t = Thread(target=main_menu, args=(termsize,))
        t.start()
        t.join()

    if config_file is not None:
        # Because the window has been initialized we need to assume that the config file is a valid path
        config_path: Optional[Path] = None
        if ".toml" in config_file:
            config_path = Path(config_file)
        else:
            config_path = config_dir / (config_file + ".toml")
        
        res = handle_file_load(window, left_bound + 10, termsize, config_path)
        if res == "exit":
            exit()
        if res is None:
            return restart()

    elif cmd == "New Configuration":
        new_config_file = new_config(window)
        if new_config_file is None or new_config_file == "":
            restart()
            return
        
        config_path = config_dir / (new_config_file + ".toml")
        config_path.touch()
        res = handle_file_load(window, left_bound + 10, termsize, config_path)
        if res == "exit":
            exit()
        if res is None:
            return restart()

    if cmd == "Load Configuration":
        window.remove_all()
        window.paint()
        configs = get_config_files(config_dir)
        res = select_option(
            (left_bound + 2, 0), configs, TermText("Select Configuration"), True
        )
        if res is None:
            return restart()
        config_file, cmd = res
        config_path = config_dir / (config_file + ".toml")
        if cmd == 0:
            res = handle_file_load(window, left_bound, termsize, config_path)
            if res == "exit":
                exit()
            if res is None:
                return restart()
        if cmd == 1:
            res = prompt_bool(TermText(f"Delete {config_file}?"), (left_bound + 2, 0))
            if res:
                os.remove(str(config_path))
            return restart()
