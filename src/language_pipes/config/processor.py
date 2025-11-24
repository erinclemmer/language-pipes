from typing import List

class HostedModel:
    id: str
    device: str
    max_memory: float # in giga bytes
    load_ends: bool # Loads head and embed of model

    def __init__(self, id: str, device: str, max_memory: float, load_ends: bool):
        self.id = id
        self.device = device
        self.max_memory = max_memory
        self.load_ends = load_ends

    @staticmethod
    def from_dict(data):
        return HostedModel(data['id'], data['device'], data['max_memory'], data['load_ends'])

class ProcessorConfig:
    model_validation: bool
    ecdsa_verification: bool
    max_pipes: int
    job_port: int
    hosted_models: List[HostedModel]

    @staticmethod
    def from_dict(data: dict) -> "ProcessorConfig":
        config = ProcessorConfig()
        config.model_validation = data['model_validation']
        config.ecdsa_verification = data['ecdsa_verification']
        config.max_pipes = data['max_pipes']
        config.job_port = data['job_port']
        config.hosted_models = [HostedModel.from_dict(o) for o in data['hosted_models']]
        return config
    
