from pathlib import Path
from typing import List, Optional

import toml


class JobProvider:
    def __init__(self):
        pass

    @staticmethod
    def get_oai_port(config_file: Path) -> Optional[int]:
        with open(config_file, 'r') as f:
            data = toml.load(f)
        
        return data.get('oai_port', None)
    
    @staticmethod
    def set_oai_port(config_file: Path, port: int):
        with open(config_file, 'r') as f:
            data = toml.load(f)

        data['oai_port'] = port

        with open(config_file, 'w') as f:
            toml.dump(data, f)

    @staticmethod
    def get_api_keys(config_file: Path) -> List[str]:
        with open(config_file, 'r') as f:
            data = toml.load(f)

        return data.get('api_keys', [])
    
    @staticmethod
    def set_api_keys(config_file: Path, keys: List[str]):
        with open(config_file, 'r') as f:
            data = toml.load(f)
            
        data['api_keys'] = keys

        with open(config_file, 'w') as f:
            toml.dump(data, f)