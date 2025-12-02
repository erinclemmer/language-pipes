from typing import Dict, Optional
from dataclasses import dataclass

from language_pipes.config.processor import ProcessorConfig
from distributed_state_network import DSNodeConfig

@dataclass
class LpConfig:
    logging_level: str
    oai_port: Optional[int]
    router: DSNodeConfig
    app_dir: str
    processor: ProcessorConfig

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        return LpConfig(
            logging_level=data['logging_level'], 
            oai_port=data['oai_port'],
            app_dir=data['app_dir'],
            router=DSNodeConfig.from_dict(data['router']), 
            processor=ProcessorConfig.from_dict(data['processor'])
        )