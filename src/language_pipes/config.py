from typing import Dict, List, Optional
from dataclasses import dataclass

from distributed_state_network import DSNodeConfig

@dataclass
class HostedModel:
    id: str
    device: str
    max_memory: float  # in gigabytes
    load_ends: bool  # Loads head and embed of model

    @staticmethod
    def from_dict(data: Dict) -> 'HostedModel':
        return HostedModel(
            id=data['id'],
            device=data['device'],
            max_memory=data['max_memory'],
            load_ends=data.get('load_ends', False)
        )

@dataclass
class LpConfig:
    # Core settings
    logging_level: str
    app_dir: str
    
    # API server
    oai_port: Optional[int]
    
    # Network configuration (external DSNodeConfig)
    router: DSNodeConfig
    
    # Model hosting
    hosted_models: List[HostedModel]
    
    # Processing options
    job_port: int
    max_pipes: int
    model_validation: bool
    ecdsa_verification: bool
    print_times: bool
    print_job_data: bool
    prefill_chunk_size: int

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        return LpConfig(
            # Core settings
            logging_level=data['logging_level'],
            app_dir=data['app_dir'],
            
            # API server
            oai_port=data.get('oai_port'),
            
            # Network configuration
            router=DSNodeConfig.from_dict(data['router']),
            
            # Model hosting
            hosted_models=[HostedModel.from_dict(m) for m in data['hosted_models']],
            
            # Processing options
            job_port=data.get('job_port', 5050),
            max_pipes=data.get('max_pipes', 1),
            model_validation=data.get('model_validation', False),
            ecdsa_verification=data.get('ecdsa_verification', False),
            print_times=data.get('print_times', False),
            print_job_data=data.get('print_job_data', False),
            prefill_chunk_size=data.get('prefill_chunk_size', 6)
        )
