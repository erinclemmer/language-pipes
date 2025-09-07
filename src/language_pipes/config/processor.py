from typing import List

class HostedModel:
    id: str
    device: str
    max_memory: float # in GigaBytes

    def __init__(self, id: str, device: str, max_memory: float):
        self.id = id
        self.device = device
        self.max_memory = max_memory

    @staticmethod
    def from_dict(data):
        return HostedModel(data['id'], data['device'], data['max_memory'])

class ProcessorConfig:
    https: bool
    job_port: int
    communication_dtype: str
    process_dtype: str
    hosted_models: List[HostedModel]

    @staticmethod
    def from_dict(data: dict) -> "ProcessorConfig":
        config = ProcessorConfig()
        config.https = data['https']
        config.job_port = data['job_port']
        config.communication_dtype['communication_dtype']
        config.process_dtype['process_dtype']
        config.hosted_models = [HostedModel.from_dict(o) for o in data['hosted_models']]
        return config
    
