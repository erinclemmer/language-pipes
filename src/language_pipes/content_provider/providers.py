from pathlib import Path

def get_providers(config_file: Path):
    from language_pipes.content_provider.provider_calls import ProviderCall
    from language_pipes.content_provider.model_provider import ModelProvider
    from language_pipes.content_provider.content_provider import ContentProvider
    from language_pipes.content_provider.network_provider import NetworkProvider
    content_provider = ContentProvider()
    
    return {
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
        ProviderCall.reset_router_logs: content_provider.network_provider.reset_router_logs,
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

        ProviderCall.host_layer_model: content_provider.model_provider.host_model,
        ProviderCall.get_layer_models: lambda: ModelProvider.get_layer_models(
            config_file
        ),
        ProviderCall.save_layer_models: lambda m: ModelProvider.save_layer_models(
            config_file, m
        ),
        ProviderCall.get_end_models: lambda: ModelProvider.get_end_models(config_file),
        ProviderCall.save_end_models: lambda em: ModelProvider.save_end_models(config_file, em),
        ProviderCall.validate_device_name: ModelProvider.validate_device_name,
        ProviderCall.get_models_status: content_provider.model_provider.get_models_status,
        ProviderCall.shutdown_models: content_provider.model_provider.shutdown_layer_models,

        
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