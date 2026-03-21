from enum import Enum

class ProviderCall(Enum):
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

    list_models = "list_models"
    get_network_status = "get_network_status"
    save_model_assignments = "save_model_assignments"
