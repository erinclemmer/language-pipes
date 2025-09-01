from typing import Dict, Optional
from dataclasses import dataclass

from language_pipes.config.processor import ProcessorConfig
from distributed_state_network import DSNodeConfig

@dataclass
class LpConfig:
    logging_level: str
    oai_port: Optional[int]
    router: DSNodeConfig
    processor: ProcessorConfig

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        return LpConfig(
            logging_level=data['logging_level'], 
            oai_port=data['oai_port'],
            router=DSNodeConfig.from_dict(data['router']), 
            processor=ProcessorConfig.from_dict(data['processor'])
        )