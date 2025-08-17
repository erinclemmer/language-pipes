import os
import json
from typing import List

class HostedModel:
    id: str
    device: str
    max_memory: int # in giga bytes

    def __init__(self, id: str, device: str, max_memory: int):
        self.id = id
        self.device = device
        self.max_memory = max_memory

    @staticmethod
    def from_json(data):
        # memory is stored as GB in the config file
        return HostedModel(data['id'], data['device'], data['max_memory'] * 10**9)

class ProcessorConfig:
    https: bool
    job_port: int
    hosted_models: List[HostedModel]

    @staticmethod
    def from_file(file: str) -> "ProcessorConfig":
        if not os.path.exists(file):
            raise FileNotFoundError(f'Config file {file} not found')
        config = ProcessorConfig()
        with open(file) as f:
            data = json.load(f)
            config.https = data['https']
            config.job_port = data['job_receive_port']
            config.hosted_models = [HostedModel.from_json(o) for o in data['hosted_models']]
        return config

    @staticmethod
    def from_dict(data: dict) -> "ProcessorConfig":
        config = ProcessorConfig()
        config.https = data['https']
        config.job_port = data['job_port']
        config.oai_port = data['oai_port'] if 'oai_port' in data else None
        config.hosted_models = [HostedModel.from_json(o) for o in data['hosted_models']]
        return config
    
