from enum import Enum


class ProviderCall(Enum):
    # Network / Configure
    get_network_config = "get_network_config"
    save_network_config = "save_network_config"

    start_network = "start_network"
    stop_network = "stop_network"
    list_peers = "list_peers"

    get_registered_node_ids = "get_registered_node_ids"
    delete_node_id = "delete_node_id"
    save_new_node_id = "save_new_node_id"

    generate_aes_key = "generate_aes_key"
    validate_aes_key = "validate_aes_key"

    detect_network_ip = "detect_network_ip"
    get_network_status = "get_network_status"
    get_total_system_ram = "get_total_system_ram"
    get_used_system_ram = "get_used_system_ram"

    # Models / Installed
    get_installed_models = "get_installed_models"
    delete_installed_model = "delete_installed_moodel"
    start_download = "start_download"
    stop_model_download = "stop_model_download"
    check_download_progress = "check_download_progress"
    download_model = "download_model"
    get_hf_token = "get_hf_token"
    save_hf_token = "save_hf_token"
    get_model_manager_logs = "get_model_manager_logs"

    # Models / Hosted
    get_models_to_load = "get_models_to_load"
    save_models_to_load = "save_models_to_load"
    validate_device_name = "validate_device_name"
    get_models_status = "get_models_status"
    host_model = "host_model"
    shutdown_models = "shutdown_models"

    # Pipes
    get_pipes_connected = "get_pipes_connected"
    get_network_pipes = "get_network_pipes"

    # Jobs
    start_oai_server = "start_oai_server"
    stop_oai_server = "stop_oai_server"
    oai_server_running = "oai_server_running"
    get_oai_logs = "get_oai_logs"

    get_oai_port = "get_oai_port"
    set_oai_port = "set_oai_port"
    get_api_keys = "get_api_keys"
    set_api_keys = "set_api_keys"

    get_active_jobs = "get_active_jobs"

    # Other
    list_models = "list_models"
    save_model_assignments = "save_model_assignments"
