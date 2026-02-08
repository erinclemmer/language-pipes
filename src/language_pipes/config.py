import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


def get_env_config() -> Dict[str, Optional[str]]:
    return {
        "logging_level": os.getenv("LP_LOGGING_LEVEL"),
        "oai_port": os.getenv("LP_OAI_PORT"),
        "app_dir": os.getenv("LP_APP_DIR"),
        "node_id": os.getenv("LP_NODE_ID"),
        "peer_port": os.getenv("LP_PEER_PORT"),
        "network_ip": os.getenv("LP_NETWORK_IP"),
        "bootstrap_address": os.getenv("LP_BOOTSTRAP_ADDRESS"),
        "bootstrap_port": os.getenv("LP_BOOTSTRAP_PORT"),
        "network_key": os.getenv("LP_NETWORK_KEY"),
        "model_validation": os.getenv("LP_MODEL_VALIDATION"),
        "max_pipes": os.getenv("LP_MAX_PIPES"),
        "layer_models": os.getenv("LP_LAYER_MODELS"),
        "prefill_chunk_size": os.getenv("LP_PREFILL_CHUNK_SIZE"),
        "num_local_layers": os.getenv("LP_NUM_LOCAL_LAYERS"),
        "model_dir": os.getenv("LP_MODEL_DIR"),
        "huggingface_token": os.getenv("LP_HUGGINGFACE_TOKEN"),
    }


def apply_env_overrides(data: Dict[str, Any], cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env_map = get_env_config()
    cli_args = cli_args or {}
    
    def precedence(key: str, cli_key: Optional[str] = None) -> Any:
        """Get value with precedence: CLI > env > config file."""
        # Check CLI args first
        arg_key = cli_key if cli_key else key
        if arg_key in cli_args and cli_args[arg_key] is not None:
            return cli_args[arg_key]
        # Then environment variables
        if key in env_map and env_map[key] is not None:
            return env_map[key]
        # Finally config file
        if key in data:
            return data[key]
        return None

    config = {
        "logging_level": precedence("logging_level"),
        "oai_port": precedence("oai_port", "openai_port"),
        "app_dir": precedence("app_dir"),
        "node_id": precedence("node_id"),
        "peer_port": precedence("peer_port"),
        "network_ip": precedence("network_ip"),
        "end_models": precedence("end_models"),
        "bootstrap_address": precedence("bootstrap_address"),
        "bootstrap_port": precedence("bootstrap_port"),
        "network_key": precedence("network_key"),
        "model_validation": precedence("model_validation"),
        "max_pipes": precedence("max_pipes"),
        "num_local_layers": precedence("num_local_layers"),
        "layer_models": precedence("layer_models"),
        "prefill_chunk_size": precedence("prefill_chunk_size"),
        "model_dir": precedence("model_dir"),
        "huggingface_token": precedence("huggingface_token"),
    }

    if config["peer_port"] is not None:
        config["peer_port"] = int(config["peer_port"])
    if config["oai_port"] is not None:
        config["oai_port"] = int(config["oai_port"])
    if config["bootstrap_port"] is not None:
        config["bootstrap_port"] = int(config["bootstrap_port"])

    if config["layer_models"] is not None:
        config["layer_models"] = parse_layer_models(config["layer_models"])

    return config


def parse_layer_models(layer_models: str | Dict) -> List[Dict[str, Any]]:
    if isinstance(layer_models, str):
        layer_models = [layer_models]
    
    result = []
    for m in layer_models:
        if isinstance(m, str):
            model_config = {}
            for pair in m.split(","):
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid format '{pair}' in '{m}'. "
                        "Expected key=value pairs (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4)"
                    )
                key, value = pair.split("=", 1)
                model_config[key.strip()] = value.strip()
            
            required_keys = {"id", "device", "memory"}
            missing = required_keys - set(model_config.keys())
            if missing:
                raise ValueError(f"Missing required keys {missing} in '{m}'")
            
            result.append({
                "id": model_config["id"],
                "device": model_config["device"],
                "max_memory": float(model_config["memory"])
            })
        else:
            result.append(m)
    
    return result

def default_config_dir() -> str:
    return str(Path.home() / ".config" / "language_pipes")


def default_model_dir() -> str:
    return str(Path.home() / ".cache" / "language_pipes" / "models")


@dataclass
class LayerModel:
    id: str
    device: str
    max_memory: float  # in gigabytes

    @staticmethod
    def from_dict(data: Dict) -> 'LayerModel':
        return LayerModel(
            id=data['id'],
            device=data['device'],
            max_memory=float(data['max_memory'])
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
    layer_models: List[LayerModel]
    end_models: List[str]
    huggingface_token: Optional[str]
    
    # Processing options
    max_pipes: int
    model_validation: bool
    prefill_chunk_size: int
    num_local_layers: int

    @staticmethod
    def from_dict(data: Dict) -> 'LpConfig':
        if data.get("node_id") is None:
            raise Exception("Node ID must be supplied to config")
        
        layer_models = data.get("layer_models", None)
        if data.get("layer_models") is None:
            layer_models = []
        
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
        
        num_local_layers = data.get('num_local_layers', None)
        if num_local_layers is None:
            num_local_layers = 1

        end_models = data.get('end_models', None)
        if end_models is None:
            end_models = []

        return LpConfig(
            # Core settings
            node_id=data.get('node_id'),
            logging_level=logging_level,
            app_dir=app_dir,
            model_dir=model_dir,
            
            # API server
            oai_port=data.get('oai_port', None),
            
            # Model hosting
            layer_models=[LayerModel.from_dict(m) for m in layer_models],
            end_models=end_models,
            huggingface_token=data.get('huggingface_token', None),
            
            # Processing options
            max_pipes=max_pipes,
            model_validation=model_validation,
            prefill_chunk_size=prefill_chunk_size,
            num_local_layers=num_local_layers
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
        lines.append(f"  {'Num local layers:':<18} {self.num_local_layers}")
        
        # Model hosting
        lines.append("")
        lines.append("--- Model Hosting ---")
        lines.append(f"  {'HuggingFace Token:':<18} {'Set' if self.huggingface_token else 'Not set'}")
        
        # Hosted models
        lines.append("")
        lines.append(f"--- Layer Models ({len(self.layer_models)}) ---")
        for i, model in enumerate(self.layer_models):
            lines.append("")
            lines.append(f"  Model #{i+1}:")
            lines.append(f"    ID:          {model.id}")
            lines.append(f"    Device:      {model.device}")
            lines.append(f"    Max Memory:  {model.max_memory} GB")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
