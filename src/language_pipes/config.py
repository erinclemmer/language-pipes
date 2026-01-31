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
            max_memory=float(data['max_memory']),
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
    prefill_chunk_size: int

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        if data.get("node_id") is None:
            raise Exception("Node ID must be supplied to config")
        if data.get("hosted_models") is None:
            raise Exception("Hosted models must be supplied to the configuration")
        
        logging_level = data.get('logging_level', None)
        if logging_level is None:
            logging_level = "INFO"
        
        app_dir = data.get('app_dir', None)
        if app_dir is None:
            app_dir = default_config_dir()
        
        model_dir = data.get('model_dir', None)
        if model_dir is None:
            model_dir = default_model_dir()
        
        model_validation = data.get('model_validation', None)
        if model_validation is None:
            model_validation = False
        
        prefill_chunk_size = data.get('prefill_chunk_size', None)
        if prefill_chunk_size is None:
            prefill_chunk_size = 6
        
        max_pipes = data.get('max_pipes', None)
        if max_pipes is None:
            max_pipes = 1

        return LpConfig(
            # Core settings
            node_id=data.get('node_id'),
            logging_level=logging_level,
            app_dir=app_dir,
            model_dir=model_dir,
            
            # API server
            oai_port=data.get('oai_port', None),
            
            # Model hosting
            hosted_models=[HostedModel.from_dict(m) for m in data['hosted_models']],
            
            # Processing options
            max_pipes=max_pipes,
            model_validation=model_validation,
            prefill_chunk_size=prefill_chunk_size
        )

    def to_string(self) -> str:
        lines = []
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("  Configuration Details")
        lines.append("=" * 60)
        
        # Core settings
        lines.append("")
        lines.append("--- Core Settings ---")
        lines.append(f"  {'Node ID:':<18} {self.node_id}")
        lines.append(f"  {'App Directory:':<18} {self.app_dir}")
        lines.append(f"  {'Model Directory:':<18} {self.model_dir}")
        lines.append(f"  {'Logging Level:':<18} {self.logging_level}")
        
        # API settings
        lines.append("")
        lines.append("--- API Settings ---")
        if self.oai_port:
            lines.append(f"  {'OpenAI API Port:':<18} {self.oai_port}")
        else:
            lines.append("  OpenAI API:         Disabled")
        
        # Processing options
        lines.append("")
        lines.append("--- Processing Options ---")
        lines.append(f"  {'Max Pipes:':<18} {self.max_pipes}")
        lines.append(f"  {'Model Validation:':<18} {'Enabled' if self.model_validation else 'Disabled'}")
        lines.append(f"  {'Prefill Chunk Size:':<18} {self.prefill_chunk_size}")
        
        # Hosted models
        lines.append("")
        lines.append(f"--- Hosted Models ({len(self.hosted_models)}) ---")
        for i, model in enumerate(self.hosted_models):
            lines.append("")
            lines.append(f"  Model #{i+1}:")
            lines.append(f"    ID:          {model.id}")
            lines.append(f"    Device:      {model.device}")
            lines.append(f"    Max Memory:  {model.max_memory} GB")
            lines.append(f"    Load Ends:   {'Yes' if model.load_ends else 'No'}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
