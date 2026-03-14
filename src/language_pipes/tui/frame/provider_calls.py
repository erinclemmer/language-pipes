from enum import Enum

class ProviderCall(Enum):
    get_network_config = "get_network_config"
    get_network_status = "get_network_status"

    get_registered_node_ids = "get_registered_node_ids"
    delete_node_id = "delete_node_id"
    save_new_node_id = "save_new_node_id"

    save_network_config = "save_network_config"
    list_models = "list_models"
    list_peers = "list_peers"
    save_model_assignments = "save_model_assignments"
