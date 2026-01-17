from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

def default_config_dir() -> str:
    return str(Path.home() / ".config" / "language_pipes")


def default_model_dir() -> str:
    return str(Path.home() / ".cache" / "language_pipes" / "models")


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
    node_id: str
    app_dir: str
    model_dir: str
    logging_level: str
    
    # API server
    oai_port: Optional[int]
    
    # Model hosting
    hosted_models: List[HostedModel]
    
    # Processing options
    max_pipes: int
    model_validation: bool
    print_times: bool
    print_job_data: bool
    prefill_chunk_size: int

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        return LpConfig(
            # Core settings
            node_id=data.get('node_id'),
            logging_level=data['logging_level'],
            app_dir=data.get('app_dir', default_config_dir()),
            model_dir=data.get('model_dir', default_model_dir()),
            
            # API server
            oai_port=data.get('oai_port'),
            
            # Model hosting
            hosted_models=[HostedModel.from_dict(m) for m in data['hosted_models']],
            
            # Processing options
            max_pipes=data.get('max_pipes', 1),
            model_validation=data.get('model_validation', False),
            print_times=data.get('print_times', False),
            print_job_data=data.get('print_job_data', False),
            prefill_chunk_size=data.get('prefill_chunk_size', 6)
        )
