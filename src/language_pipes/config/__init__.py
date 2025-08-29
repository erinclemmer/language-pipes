import os
import toml
from typing import Dict, Optional
from dataclasses import dataclass

from language_pipes.config.processor import ProcessorConfig
from distributed_state_network import DSNodeConfig

@dataclass
class LMNetConfig:
    logging_level: str
    oai_port: Optional[int]
    router: DSNodeConfig
    processor: ProcessorConfig

    @staticmethod
    def from_dict(data: Dict) -> 'LMNetConfig':
        return LMNetConfig(
            logging_level=data['logging_level'] if 'logging_level' in data else "INFO", 
            oai_port=data['oai_port'] if 'oai_port' in data else None,
            router=DSNodeConfig.from_dict(data['router']), 
            processor=ProcessorConfig.from_dict(data['processor'])
        )