from enum import Enum

class ProviderCall(Enum):
    # Network / Configure
    get_network_config = "get_network_config"
    save_network_config = "save_network_config"

    start_network = "start_network"
    stop_network = "stop_network"

    get_router_status = "get_router_status"
    list_peers = "list_peers"

    get_registered_node_ids = "get_registered_node_ids"
    delete_node_id = "delete_node_id"
    save_new_node_id = "save_new_node_id"

    generate_aes_key = "generate_aes_key"
    validate_aes_key = "validate_aes_key"

    detect_network_ip = "detect_network_ip"
    get_network_status = "get_network_status"

    # Models / Installed
    get_installed_models = "get_installed_models"
    delete_installed_model = "delete_installed_moodel"
    start_download = "start_download"
    stop_model_download = "stop_model_download"
    check_download_progress = "check_download_progress"
    download_model = "download_model"

    # Other
    list_models = "list_models"
    save_model_assignments = "save_model_assignments"
